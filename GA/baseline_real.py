import time
import pickle
from typing import List, Tuple, Optional, Set, Dict, Callable
import os
import shutil
import argparse

from numba import njit, jit
import numpy as np
import pandas as pd

from functools import partial
import wandb

import ruck
from ruck.external.deap import select_nsga2
from ruck import R, Trainer
from ruck.external.ray import *

# from Warehouse.grid import UniformRandomGrid
from Warehouse.grid import RealWarehouse

from Logging.log import MultiObjLog

from Warehouse.grid import CONFIG
from MAPD.token_passing import TokenPassing


def evaluate(values: np.ndarray, no_agents: int, no_timesteps: int, start_locs, dropoff_locs):
    try:
        grid: List = values.tolist()
        y_len = len(grid)

        # non_task_endpoints = [(0, y) for y in range(y_len)]
        # start_locs = non_task_endpoints[:no_agents]

        # Need to swap x and y
        non_task_endpoints = [(x, y) for (y, x) in start_locs]

        start_locs = non_task_endpoints[:no_agents]

        tp = TokenPassing(grid, no_agents, start_locs, non_task_endpoints, no_timesteps, task_frequency=1,
                          is_logging_collisions=True)
        final_agents = tp.compute()
        unique_tasks_completed = tp.get_no_unique_tasks_completed()
        # print(unique_tasks_completed)
        no_unreachable_locs = tp.get_no_unreachable_locs()
        # reachable_locs = NO_LOCS - no_unreachable_locs
        reachable_locs = CONFIG["no_opt_locs"] - no_unreachable_locs

    except Exception as e:  # Bad way to do this but I do not want this to crash after 5hrs of training from a random edge case
        print(f"Exception occurred: {e}")
        tasks_completed = 0
        reachable_locs = 0
        unique_tasks_completed = 0

    return unique_tasks_completed, reachable_locs


class WarehouseGAModule(ruck.EaModule):
    def __init__(
            self,
            population_size: int = 300,
            no_agents=5,
            no_timesteps=500,
            warehouse=None,
            offspring_num: int = None,  # offspring_num (lambda) is automatically set to population_size (mu) when `None`
            member_size: int = 100,
            p_mate: float = 0.5,
            p_mutate: float = 0.5,
            ea_mode: str = 'mu_plus_lambda',
            mut_tile_no: int = 1,
            mut_tile_size: int = 5,
            log_interval: int = -1,
            save_interval: int = -1,
            no_generations: int = 0,
            pop_save_dir: str = "",
            log_folder_base_path: str = "",
            log_name: str = ""
    ):
        self._population_size = population_size
        self.save_hyperparameters()
        # self.eval_count = 0

        self.curr_gen = 0
        self.evals = 0

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.no_generations = no_generations
        self.pop_save_dir = pop_save_dir

        self.warehouse = warehouse
        if warehouse is None:
            self.warehouse = RealWarehouse()

        self.start_locs = self.warehouse.non_task_endpoints
        self.dropoff_locs = self.warehouse.dropoff_locs

        self.multi_obj_log: Optional[MultiObjLog] = None

        if log_name != "":
            self.multi_obj_log = MultiObjLog(log_folder_base_path, log_name, log_interval, save_interval,
                                             no_generations, CONFIG["no_opt_locs"])

        # self.train_loop_func = partial(train_loop_func, self)
        def _mate(arr_1: np.ndarray, arr_2: np.ndarray):
            return arr_1, arr_2
            # return mate_njit(arr_1, arr_2,
            #                  tile_size=self.mut_tile_size, tile_no=self.mut_tile_no)

        def _mutate(arr):
            return arr

        # implement the required functions for `EaModule`
        self.generate_offspring, self.select_population = R.make_ea(
            mode=self.hparams.ea_mode,
            offspring_num=self.hparams.offspring_num,
            # decorate the functions with `ray_remote_put` which automatically
            # `ray.get` arguments that are `ObjectRef`s and `ray.put`s returned results
            mate_fn=ray_remote_puts(_mate).remote,
            mutate_fn=ray_remote_put(_mutate).remote,
            # efficient to compute locally
            # select_fn=functools.partial(R.select_tournament, k=3),
            select_fn=select_nsga2,
            p_mate=self.hparams.p_mate,
            p_mutate=self.hparams.p_mutate,
            # ENABLE multiprocessing
            map_fn=ray_map,
        )

        def _eval(values):
            # Evaluating random
            values = self.warehouse.create_grid()
            return evaluate(values, no_agents, no_timesteps, self.start_locs, self.dropoff_locs)

        # eval_partial = partial()
        # eval function, we need to cache it on the class to prevent
        # multiple calls to ray.remote. We use ray.remote instead of
        # ray_remote_put like above because we want the returned values
        # not object refs to those values.
        self._ray_eval = ray.remote(_eval).remote

    def evaluate_values(self, value_refs):
        out = ray_map(self._ray_eval, value_refs)
        self.evals += len(out)
        data = None

        if self.multi_obj_log is not None:
            vals = [ray.get(val_ref) for val_ref in value_refs]
            self.multi_obj_log.log(fitnesses=out, population=vals,
                                   generation=self.curr_gen + 1, evals=self.evals)

        NO_LOCS = CONFIG["no_opt_locs"]

        if self.log_interval > -1 and (self.curr_gen == 0 or (self.curr_gen + 1) % self.log_interval == 0 or self.curr_gen + 1 == self.no_generations):
            gen = self.curr_gen+1
            data = [[x, y/NO_LOCS, gen] for (x, y) in out]
            table = wandb.Table(data=data, columns=["unique_tasks_completed", "perc_reachable_locs", "gen"])
            wandb.log({f"fitness_scatterplot": wandb.plot.scatter(table, "unique_tasks_completed", "perc_reachable_locs",
                                                                  title=f"Gen = {gen} - Unique Tasks Completed Vs Percentage of Locs Reachable")})

        # wandb logging
        if self.log_interval > -1:
            reachable_locs = [y for (x, y) in out]
            unique_tasks_completed = [x for (x, y) in out]
            log_dict = {
                "generation": self.curr_gen,
                "perc_reachable_locs_max": np.max(reachable_locs)/NO_LOCS,
                "perc_reachable_locs_mean": np.mean(reachable_locs)/NO_LOCS,
                "perc_reachable_locs_min": np.min(reachable_locs)/NO_LOCS,
                "perc_reachable_locs_var": np.var(reachable_locs)/NO_LOCS,

                "unique_tasks_completed_max": np.max(unique_tasks_completed),
                "unique_tasks_completed_mean": np.mean(unique_tasks_completed),
                "unique_tasks_completed_min": np.min(unique_tasks_completed),
                "unique_tasks_completed_var": np.var(unique_tasks_completed),
            }
            wandb.log(log_dict)
            # tbl_data = [[x, y, self.curr_gen+1] for (x, y) in data]
            if data is None:
                gen = self.curr_gen+1
                data = [[x, y/NO_LOCS, gen] for (x, y) in out]

            # Save fitnesses at each generation
            fitness_table = wandb.Table(columns=["unique_tasks_completed", "reachable_locs", "gen"], data=data)
            wandb.log({"fitness_table": fitness_table})

        if self.save_interval > -1 and (self.curr_gen == 0 or (self.curr_gen + 1) % self.save_interval == 0 or self.curr_gen + 1 == self.no_generations):
            # data = [[x, y] for (x, y) in out]
            values = [ray.get(val_ref) for val_ref in value_refs]
            val_data = list(zip(values, out))

            file_name = os.path.join(wandb.run.dir, f"pop_{self.curr_gen+1}.pkl")
            with open(file_name, "wb") as f:
                pickle.dump(val_data, f)

            wandb.save(file_name)

        self.curr_gen += 1
        return out

    def gen_starting_values(self):
        pop_size = self.hparams.population_size
        return [ray.put(self.warehouse.create_grid())
                for _ in range(pop_size)]
        # return gen_starting_points_rand(self.hparams.population_size, self.warehouse)


def train(pop_size, n_generations, n_agents,
          n_timesteps, n_cores,
          using_wandb, wandb_mode, log_interval, save_interval,
          log_folder_path, log_name,
          cluster_node,
          run_notes, run_name, tags):
    # initialize ray to use the specified system resources
    if n_cores <= 0:
        ray.init()
    else:
        ray.init(num_cpus=n_cores)

    config = {
        "pop_size": pop_size,
        "n_generations": n_generations,
        "no_agents": n_agents,
        "no_timesteps": n_timesteps,
        "fitness": "unique_tasks_completed, reachable_locs",
        "cluster_node": cluster_node,
        "mate_func": "none"
    }
    # notes = "Test to see if this works :)"
    if run_name == "":
        run_name = None

    if using_wandb:
        wandb.init(project="GARuck", entity="simonrosen42", config=config,
                   notes=run_notes, name=run_name, tags=tags, mode=wandb_mode)

        # define our custom x axis metric
        wandb.define_metric("generation")
        # define which metrics will be plotted against it
        # Against generation
        wandb.define_metric("perc_reachable_locs_max", step_metric="generation")
        wandb.define_metric("perc_reachable_locs_mean", step_metric="generation")
        wandb.define_metric("perc_reachable_locs_min", step_metric="generation")
        wandb.define_metric("perc_reachable_locs_var", step_metric="generation")

        wandb.define_metric("unique_tasks_completed_max", step_metric="generation")
        wandb.define_metric("unique_tasks_completed_mean", step_metric="generation")
        wandb.define_metric("unique_tasks_completed_min", step_metric="generation")
        wandb.define_metric("unique_tasks_completed_var", step_metric="generation")
    else:
        log_interval = -1
        save_interval = -1

    module = WarehouseGAModule(population_size=pop_size,
                               no_generations=n_generations, no_agents=n_agents,
                               no_timesteps=n_timesteps,
                               log_interval=log_interval, save_interval=save_interval,
                               log_folder_base_path=log_folder_path, log_name=log_name)
    trainer = Trainer(generations=n_generations, progress=True)
    pop, logbook, halloffame = trainer.fit(module)

    wandb.finish()


def test():
    # pop_size, n_generations, n_agents,
    #           n_timesteps, n_cores,
    #           using_wandb, wandb_mode, log_interval, save_interval,
    #           log_folder_path, log_name,
    #           cluster_node,
    #           run_notes, run_name, tags
    train(pop_size=50, n_generations=1,
          n_agents=60, n_timesteps=500, n_cores=-1,
          using_wandb=False, wandb_mode="disabled", log_interval=1, save_interval=1,
          log_folder_path="", log_name="",
          cluster_node=-1, run_notes="", run_name="Test Offline", tags=["test"])


if __name__ == "__main__":
    test()
