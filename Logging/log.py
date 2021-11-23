import os, shutil

import numpy as np
import pandas as pd
import pickle


class MultiObjLog:
    def __init__(self, folder_base_path, log_name, save_fitness_interval,
                 save_pop_interval, max_steps, no_locs):
        self.folder_base_path = folder_base_path
        self.log_folder_path = f"{folder_base_path}/{log_name}"
        self.log_name = log_name
        self.stat_file_path = f"{self.log_folder_path}/stats.csv"

        self.save_fitness_interval = save_fitness_interval
        self.save_pop_interval = save_pop_interval
        self.max_steps = max_steps
        self.no_locs = no_locs

        self.curr_step = 0

        if os.path.isdir(self.log_folder_path):
            for files in os.listdir(self.log_folder_path):
                path = os.path.join(self.log_folder_path, files)
                try:
                    shutil.rmtree(path)
                except OSError:
                    os.remove(path)
        else:
            os.mkdir(self.log_folder_path)

        os.mkdir(f"{self.log_folder_path}/pops")
        os.mkdir(f"{self.log_folder_path}/fitnesses")

        with open(f"{self.log_folder_path}/stats.csv", "w") as f:
            column_str = "generation,evals," \
                         "perc_reachable_locs_max,perc_reachable_locs_min,perc_reachable_locs_mean," \
                         "perc_reachable_locs_var," \
                         "unique_tasks_completed_max,unique_tasks_completed_min,unique_tasks_completed_mean," \
                         "unique_tasks_completed_var"
            f.write(column_str + "\n")

    def log(self, fitnesses, population, generation, evals):
        unique_tasks_completed = [x for (x, y) in fitnesses]
        perc_reachable_locs = [y/self.no_locs for (x, y) in fitnesses]

        # Save Stats
        perc_reachable_locs_max = np.max(perc_reachable_locs)
        perc_reachable_locs_min = np.min(perc_reachable_locs)
        perc_reachable_locs_mean = np.mean(perc_reachable_locs)
        perc_reachable_locs_var = np.var(perc_reachable_locs)

        unique_tasks_completed_max = np.max(unique_tasks_completed)
        unique_tasks_completed_min = np.min(unique_tasks_completed)
        unique_tasks_completed_mean = np.mean(unique_tasks_completed)
        unique_tasks_completed_var = np.var(unique_tasks_completed)

        data_str = f"{generation},{evals}," \
                   f"{perc_reachable_locs_max},{perc_reachable_locs_min},{perc_reachable_locs_mean}," \
                   f"{perc_reachable_locs_var}," \
                   f"{unique_tasks_completed_max},{unique_tasks_completed_min},{unique_tasks_completed_mean}," \
                   f"{unique_tasks_completed_var}"

        with open(self.stat_file_path, "a") as f:
            f.write(data_str+"\n")

        # Save fitnesses
        if self.curr_step % self.save_fitness_interval == 0 or self.curr_step == self.max_steps:
            df_dict = {
                "unique_tasks_completed": unique_tasks_completed,
                "perc_reachable_locs": perc_reachable_locs
            }
            csv_path = f"{self.log_folder_path}/fitnesses/gen_{generation}.csv"
            df = pd.DataFrame.from_dict(df_dict)
            df["pop"] = population
            df.to_csv(csv_path, index=False)

        # Save pops
        if self.curr_step % self.save_pop_interval == 0 or self.curr_step == self.max_steps:
            data = list(zip(fitnesses, population))
            file_name = f"{self.log_folder_path}/pops/pop_{generation}.pkl"
            with open(file_name, "wb") as f:
                pickle.dump(data, f)

        self.curr_step += 1
