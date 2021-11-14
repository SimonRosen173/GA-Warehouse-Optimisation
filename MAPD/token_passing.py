# Based off [Ma et al. 2017]
import time
from typing import Type, Tuple, List, Set, Optional, Dict

from Graph.graph import GridGraph
from Warehouse.grid import UniformRandomGrid, RealWarehouse

# from GlobalObjs.GraphNX import GridGraph, plot_graph
# from Benchmark import Warehouse
# from Cooperative_AStar.CooperativeAStar import cooperative_astar_path, man_dist
# from Visualisations.Vis import VisGrid

from MAPD.agent import Agent
from MAPD.task_assigner import TaskAssigner, Task
# from Cooperative_AStar.CooperativeAStar import cooperative_astar_path_fast

# noinspection PyUnresolvedReferences
from MAPD.cooperative_astar import cooperative_astar_path


import numpy as np
import pandas as pd
from numpy import random
from numba import njit
from itertools import chain

# random.seed(42)

Path = List[Tuple[Tuple[int, int], int]]


@njit
def man_dist(node1, node2):
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])


# NOTE: Token must contain full paths of agents not just current
class Token:
    def __init__(self, no_agents, start_locs: List[Tuple[int, int]], non_task_endpoints: List[Tuple[int, int]],
                 start_t=0):
        self._resv_rbl = set()
        self._resv_locs = set()

        self._no_agents = no_agents
        self._non_task_endpoints = non_task_endpoints

        # each path is a space-time path, i.e. ((0, 0), 1) means agent is at (0, 0) at timestep = 1
        self._paths: List[Path] = [[] for _ in range(no_agents)]
        self._all_path_ends: List[List[Tuple[int, int]]] = [[] for _ in range(no_agents)]
        # self._path_ends = [[((1, 1), 1)]]

        # Pos and time interval agents are stationary
        self._stationary_list: List[List[Tuple[Tuple, int, int]]] = [[] for _ in range(no_agents)]
        self._is_stationary_list: List[bool] = [False]*no_agents
        self._curr_stationary: List[Optional[Tuple[int, int]]] = [None for _ in range(no_agents)]

        for agent_ind in range(no_agents):
            self._paths[agent_ind] = [(start_locs[agent_ind], start_t)]
            self.add_stationary(agent_ind, start_locs[agent_ind], start_t)

        # self.resv_tbl = set()
        # self.resv_locs = set()
        # self.path_end_locs = set()
        pass

    def add_stationary(self, agent_id: int, pos: Tuple[int, int], start_t: int):
        if not self._is_stationary_list[agent_id]:
            # add_stationary sets agent specified as stationary
            self._is_stationary_list[agent_id] = True
            self._curr_stationary[agent_id] = pos
            self._stationary_list[agent_id].append((pos, start_t, np.inf))

    # pos only required for error checking
    def _update_last_end_t(self, agent_id: int, pos: Tuple[int, int], end_t: int):
        last_stationary = self._stationary_list[agent_id][-1]
        assert last_stationary[0] == pos, f"{last_stationary[0]} != {pos} - pos specified must correspond to last element in _stationary_list[agent_id]"
        last_stationary = (last_stationary[0], last_stationary[1], end_t)
        self._stationary_list[agent_id][-1] = last_stationary

    def add_to_path(self, agent_id: int, path: List[Tuple[Tuple[int, int], int]]):
        # If agent was stationary set end time of relevant element of _stationary_list
        if self._is_stationary_list[agent_id]:  # and len(self._paths[agent_id]) > 0 and len(self._stationary_list[agent_id]) > 0
            end_t = self._paths[agent_id][-1][1]
            last_stationary = self._stationary_list[agent_id][-1]
            last_stationary = (last_stationary[0], last_stationary[1], end_t)
            self._stationary_list[agent_id][-1] = last_stationary

        # add_to_path sets agent specified as not stationary
        self._is_stationary_list[agent_id] = False

        self._paths[agent_id].extend(path)

    # Probs make this neater later...
    def set_path_ends(self, agent_id: int, path_ends: List[Tuple[int, int]]):
        self._all_path_ends[agent_id] = path_ends

    def is_stationary(self, agent_id: int) -> bool:
        return self._is_stationary_list[agent_id]

    # def update_path(self, agent_ind: int, path: List[Tuple[Tuple[int, int], int]], is_stationary):
    #     self._paths[agent_ind] = path
    #     self._is_stationary_list[agent_ind] = is_stationary

    # Does not work well
    def has_path_ending_in(self, locs: List[Tuple[int, int]]) -> bool:
        # for path in self._paths:
        #     if len(path) > 0 and path[-1][0] in locs:
        #         return True
        for path_ends in self._all_path_ends:
            for loc in locs:
                if loc in path_ends:
                    return True

        for stat in self._stationary_list:
            last_stationary = stat[-1]
            if last_stationary[0] in locs:
                return True

        return False

    # Is there a non-task endpoint at pos excluding the non-task endpoint of specified agent
    def non_task_endpoint_at(self, pos: Tuple[int, int], agent_id):
        for curr_agent_id in range(self._no_agents):
            if curr_agent_id != agent_id and self._non_task_endpoints[curr_agent_id] == pos:
                return True
        return False

    # Reservation table for agent given should not include the prev path of that agent
    def get_resv_tbl(self, agent_id: int, curr_t: int) -> Set[Tuple[Tuple[int, int], int]]:
        resv_tbl: Set[Tuple[Tuple[int, int], int]] = set()
        # resv_tbl = self._resv_rbl
        for curr_agent_id in range(self._no_agents):
            if curr_agent_id != agent_id:  # and not self._is_stationary_list[curr_agent_id]
                for el in self._paths[curr_agent_id]:
                    if el[1] >= curr_t:  # NEW
                        resv_tbl.add(el)
                        # Reserve at next time step to avoid head-on/pass-through collisions
                        resv_tbl.add((el[0], el[1]+1))
                # Add space time points where relevant agent was stationary
                for stationary_loc in self._stationary_list[curr_agent_id]:
                    if stationary_loc[2] >= curr_t and stationary_loc[2] != np.inf:
                        loc = stationary_loc[0]
                        space_time_locs = [(loc, t) for t in range(stationary_loc[2], stationary_loc[3]+2)]
                        resv_tbl.update(space_time_locs)
        return resv_tbl

    def get_resv_locs(self, agent_id: int, curr_t: int) -> Set[Tuple[int, int]]:
        resv_locs: Set[Tuple[int, int]] = set()
        for curr_agent_id in range(self._no_agents):
            if curr_agent_id != agent_id:
                if self._is_stationary_list[curr_agent_id]:
                    resv_locs.add(self._curr_stationary[curr_agent_id])
                resv_locs.add(self._non_task_endpoints[curr_agent_id])
        return resv_locs

    def get_resv_locs_old(self, agent_id: int, curr_t: int) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        resv_locs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for curr_agent_id in range(self._no_agents):
            if curr_agent_id != agent_id:
                for stationary_loc in self._stationary_list[curr_agent_id]:
                    if stationary_loc[2] >= curr_t:  # NEW
                        if stationary_loc[0] not in resv_locs:
                            resv_locs[stationary_loc[0]] = [(stationary_loc[1], stationary_loc[2])]
                        else:
                            resv_locs[stationary_loc[0]].append((stationary_loc[1], stationary_loc[2]))
                resv_locs[self._non_task_endpoints[curr_agent_id]] = [(0, np.inf)]

        return resv_locs

    def get_last_path_locs(self):
        return [path[-1] for path in self._paths]

    def get_agents_with(self, loc):
        ids = []
        for agent_id in range(self._no_agents):
            if loc in self._paths[agent_id]:
                ids.append(agent_id)
        return ids

    # # Reserved locations for agent given should not include the prev path of that agent
    # def get_resv_locs(self, agent_id: int) -> Dict[Tuple[int, int], int]: # -> List[Tuple[Tuple[int, int], int]]:
    #     # resv_locs: List[Tuple[Tuple[int, int], int]] = []
    #     resv_locs: Dict[Tuple[int, int], int] = {}
    #     for curr_agent_id in range(self._no_agents):
    #         if curr_agent_id != agent_id and self._is_stationary_list[curr_agent_id]:
    #             resv_locs[self._paths[curr_agent_id][-1][0]] = self._paths[curr_agent_id][-1][1]
    #     return resv_locs


class TokenPassing:
    def __init__(self, grid, no_agents, start_locs, non_task_endpoints, max_t, unreachable_locs=None,
                 task_frequency=1, start_t=0, is_logging_collisions=False):
        self._no_agents = no_agents
        self._grid = grid
        self._np_grid = np.array(grid, dtype=int)
        self._max_t = max_t
        self._non_task_endpoints: List[Tuple[int, int]] = non_task_endpoints
        self._is_logging_collisions = is_logging_collisions

        self.time_elapsed: float = 0.0
        self.time_elapsed_c_astar: float = 0.0
        self.coop_astar_calls: int = 0

        grid_graph = GridGraph(grid, only_full_G=True)
        grid_graph.remove_non_reachable()
        self._graph = grid_graph.get_full_G()

        self._agents = [Agent(i, start_locs[i]) for i in range(no_agents)]

        self._token = Token(no_agents, start_locs, non_task_endpoints)

        # self._token.update_path(agent_ind, [(start_locs[agent_ind], start_t)], True)
        # for start_loc in start_locs:
        #     self._token.update_path()
        #     self._token.resv_locs.add(start_loc)

        yx_unreachable_locs = grid_graph.get_unreachable_nodes()
        xy_unreachable_locs = [(x, y) for (y, x) in yx_unreachable_locs]
        self._unreachable_locs = set(xy_unreachable_locs)
        # if unreachable_locs is None:
        #     pass

        self._ta = TaskAssigner(grid, self._unreachable_locs, task_frequency)

    def compute(self):
        token = self._token
        agents = self._agents
        graph = self._graph
        ta = self._ta
        max_t = self._max_t
        np_grid = self._np_grid

        curr_t = 0

        start_time_all = time.time()

        while curr_t < max_t:
            for agent in agents:
                # Skip agent if not ready
                if not agent.is_ready():
                    agent.inc_timestep()
                    continue

                # Do task stuff
                tasks = ta.get_ready_tasks()
                # I.e. task set prime in pseudo code
                tasks_prime = []
                for task in tasks:
                    # if not (task.pickup_point in token.path_end_locs or task.dropoff_point in token.path_end_locs):
                    if not token.has_path_ending_in([task.pickup_point, task.dropoff_point]):
                        tasks_prime.append(task)

                if len(tasks_prime) > 0:
                    # get task with pickup point closest to curr loc of agent
                    min_dist = np.inf
                    min_task: Optional[Task] = None
                    for task in tasks_prime:
                        curr_dist = man_dist(agent.curr_loc, task.pickup_point)
                        if curr_dist < min_dist:
                            min_task = task

                    resv_tbl = token.get_resv_tbl(agent.id, curr_t)
                    resv_locs = token.get_resv_locs(agent.id, curr_t)

                    ta.remove_task_from_ready(min_task)

                    start_time = time.time()

                    # noinspection PyTypeChecker
                    path = cooperative_astar_path(np_grid, source=agent.curr_loc, target=min_task.pickup_point,
                                                          resv_tbl=resv_tbl, resv_locs=resv_locs, start_t=curr_t)
                    # paths, _ = cooperative_astar_path(graph, [agent.curr_loc], [min_task.pickup_point], resv_tbl=resv_tbl,
                    #                                   resv_locs=resv_locs, start_t=curr_t)
                    self.time_elapsed_c_astar += time.time() - start_time

                    path_to_pickup = path
                    # path_to_pickup = paths[0]
                    pickup_t = path_to_pickup[-1][1]

                    start_time = time.time()
                    # noinspection PyTypeChecker
                    path = cooperative_astar_path(np_grid, source=min_task.pickup_point, target=min_task.dropoff_point,
                                                  resv_tbl=resv_tbl, resv_locs=resv_locs, start_t=pickup_t)
                    # paths, _ = cooperative_astar_path(graph, [min_task.pickup_point], [min_task.dropoff_point], resv_tbl=resv_tbl,
                    #                                   resv_locs=resv_locs, start_t=pickup_t)
                    self.time_elapsed_c_astar += time.time() - start_time
                    self.coop_astar_calls += 2

                    path_to_dropoff = path
                    # path_to_dropoff = paths[0]

                    agent.assign_task(min_task, path_to_pickup, path_to_dropoff)

                    path = path_to_pickup + path_to_dropoff[1:]

                    token.add_to_path(agent.id, path)
                    token.set_path_ends(agent.id, [min_task.pickup_point, min_task.dropoff_point])
                    # token.update_path(agent.id, path, False)
                else:
                    is_stationary_valid = True  # if no task has goal at agent's current location
                    for task in tasks:
                        if task.dropoff_point == agent.curr_loc:
                            is_stationary_valid = False
                            break

                    if is_stationary_valid:
                        # Update agent's path in token with stationary path
                        token.add_stationary(agent.id, agent.curr_loc, curr_t)
                        # token.update_path(agent.id, [(agent.curr_loc, curr_t)], True)

                    else:
                        goal = None
                        # Update agent's path in token with deadlock avoidance path
                        # i.e. path to non-occupied non-task endpoint
                        # for endpoint in self._non_task_endpoints:
                        #     if not token.has_path_ending_in([endpoint]):
                        #         goal = endpoint
                        #         break
                        # if goal is None:
                        #     raise Exception("No valid non-task endpoints found :(")
                        goal = self._non_task_endpoints[agent.id]  # Each agent has unique non-task endpoint

                        # find path to goal and update agent & token
                        source = agent.curr_loc
                        resv_tbl = token.get_resv_tbl(agent.id, curr_t)
                        resv_locs = token.get_resv_locs(agent.id, curr_t)

                        # if agent.id == 7 and curr_t == 265:
                        #     last_locs = token.get_last_path_locs()
                        #     tmp = ((11, 2), 265) in resv_tbl
                        #     tmp_1 = ((11, 2), 264) in resv_tbl
                        #     tmp_3 = ((11, 2), 266) in resv_tbl
                        #     agents_ids = token.get_agents_with(((11, 2), 265))

                        start_time = time.time()

                        # noinspection PyTypeChecker
                        path = cooperative_astar_path(np_grid, source=source, target=goal,
                                                      resv_tbl=resv_tbl, resv_locs=resv_locs, start_t=curr_t)

                        # paths, _ = cooperative_astar_path(graph, [source], [goal], resv_tbl=resv_tbl, resv_locs=resv_locs, start_t=curr_t)
                        self.time_elapsed_c_astar += time.time() - start_time
                        self.coop_astar_calls += 1

                        # path = paths[0]
                        agent.assign_avoidance_path(path)
                        # token.update_path(agent.id, path, False)
                        token.add_to_path(agent.id, path)
                        token.set_path_ends(agent.id, [goal])

                agent.inc_timestep()

            if self._is_logging_collisions:
                for agent_id_1 in range(self._no_agents):
                    for agent_id_2 in range(agent_id_1 + 1, self._no_agents):
                        if self._agents[agent_id_1].curr_loc == self._agents[agent_id_2].curr_loc:
                            collide_agent_1 = self._agents[agent_id_1]
                            collide_agent_2 = self._agents[agent_id_2]
                            print(f"COLLISION(t = {curr_t}) - collision between agent {agent_id_1} and agent {agent_id_2} at "
                                  f"pos = {self._agents[agent_id_1].curr_loc}")
                pass

            curr_t += 1
            ta.inc_timestep()

        self.time_elapsed = time.time() - start_time_all
        return agents

    def get_no_tasks_completed(self) -> int:
        return sum([agent.get_no_tasks_completed() for agent in self._agents])

    def get_no_unique_tasks_completed(self) -> int:
        task_pickups = [agent.get_tasks_completed_pickups() for agent in self._agents]
        task_pickups = list(chain(*task_pickups))
        # print(task_pickups)
        values, counts = np.unique(task_pickups, return_counts=True)
        # print(f"Not unique length: {len(task_pickups)} \t Is unique length: {len(values)}")
        return len(values)

    def get_no_unreachable_locs(self) -> int:
        return len(self._unreachable_locs)


def visualise_paths(grid, agents: List[Agent]):
    full_paths = []
    for agent in agents:
        full_path = agent.get_full_path()
        full_paths.append(full_path)

    max_len = max(len(arr) for arr in full_paths)
    for i, path in enumerate(full_paths):
        if len(path) < max_len:
            end_path = [path[-1]] * (max_len - len(path))
            full_paths[i].extend(end_path)

    print("SENSE CHECKING:")
    for pos_ind in range(max_len):
        for agent_ind_1 in range(len(full_paths)):
            for agent_ind_2 in range(agent_ind_1+1, len(full_paths)):
                if full_paths[agent_ind_1][pos_ind] == full_paths[agent_ind_2][pos_ind]:
                    print(f"Collision between {agent_ind_1} and {agent_ind_2} at t = {pos_ind}, "
                          f"pos = {full_paths[agent_ind_1][pos_ind]}")

    new_vis = VisGrid(grid, (800, 400), 25, tick_time=0.2)
    new_vis.window.getMouse()
    new_vis.animate_multi_path(full_paths, is_pos_xy=False)
    # new_vis.animate_path(full_paths_dict[0], is_pos_xy=False)
    new_vis.window.getMouse()


# def benchmark_tp_no_agents():
#     no_agents_arr = [1, 2, 5, 10, 15, 20]
#     n = len(no_agents_arr)
#     max_t = 500
#     no_runs = 20
#
#     # print("##############")
#     # print("REAL WAREHOUSE")
#     # print("##############")
#     #
#     # # Real Warehouse Layout
#     # grid = Warehouse.txt_to_grid("map_warehouse_1.txt", use_curr_workspace=True, simple_layout=False)
#     # y_len = len(grid)
#     # non_task_endpoints = [(0, y) for y in range(y_len)]
#     #
#     # # Time Taken Vs No of Agents
#     #
#     # times_taken_avg = [0.0] * n
#     # tasks_completed_avg = [0.0] * n
#     # times_taken_var = [0.0] * n
#     # tasks_completed_var = [0.0] * n
#     #
#     # for i, no_agents in enumerate(no_agents_arr):
#     #     print(f"\nno_agents = {no_agents}")
#     #     start_locs = non_task_endpoints[:no_agents]
#     #     curr_tasks_completed = []
#     #     curr_times_taken = []
#     #
#     #     for run_no in range(no_runs):
#     #         print(f"\trun_no={run_no} - ", end="")
#     #         start = time.time()
#     #         tp = TokenPassing(grid, no_agents, start_locs, non_task_endpoints, max_t, task_frequency=1,
#     #                           is_logging_collisions=False)
#     #         final_agents = tp.compute()
#     #         time_elapsed = time.time() - start
#     #         curr_tasks_completed.append(tp.get_no_tasks_completed())
#     #         curr_times_taken.append(time_elapsed)
#     #         print(f"time_elapsed: {time_elapsed:.4f}")
#     #
#     #         # times_taken_avg[i] += time_elapsed
#     #
#     #     times_taken_avg[i] = np.mean(curr_times_taken)
#     #     tasks_completed_avg[i] = np.mean(curr_tasks_completed)
#     #     times_taken_var[i] = np.var(curr_times_taken)
#     #     tasks_completed_var[i] = np.var(curr_tasks_completed)
#     #     print(f"\ttotal time_elapsed {sum(curr_times_taken): .4f}")
#     # df_dict = {
#     #     "no_agents": no_agents_arr,
#     #     "time_taken_avg": times_taken_avg,
#     #     "time_taken_var": times_taken_var,
#     #     "tasks_completed_avg": tasks_completed_avg,
#     #     "tasks_completed_var": tasks_completed_var
#     # }
#     # df = pd.DataFrame.from_dict(df_dict)
#     # df.to_csv("benchmarks/no_agents_vs_t_tc_real.csv", index=False)
#
#     print("##############")
#     print("RAND WAREHOUSE")
#     print("##############")
#     num_storage_locs = 560  # 560
#     storage_shape = (22, 44)
#
#     # Time Taken Vs No of Agents
#
#     times_taken_avg = [0.0] * n
#     tasks_completed_avg = [0.0] * n
#     times_taken_var = [0.0] * n
#     tasks_completed_var = [0.0] * n
#
#     for i, no_agents in enumerate(no_agents_arr):
#         print(f"\nno_agents = {no_agents}")
#         curr_tasks_completed = []
#         curr_times_taken = []
#
#         for run_no in range(no_runs):
#             print(f"\trun_no={run_no} - generating grid... ", end="")
#             grid = Warehouse.get_uniform_random_grid(storage_shape, num_storage_locs)
#             y_len = len(grid)
#             non_task_endpoints = [(0, y) for y in range(y_len)]
#             start_locs = non_task_endpoints[:no_agents]
#
#             print(f"doing MAPD... ", end="")
#             start = time.time()
#             tp = TokenPassing(grid, no_agents, start_locs, non_task_endpoints, max_t, task_frequency=1,
#                               is_logging_collisions=False)
#             final_agents = tp.compute()
#             time_elapsed = time.time() - start
#             curr_tasks_completed.append(tp.get_no_tasks_completed())
#             curr_times_taken.append(time_elapsed)
#             print(f"time_elapsed: {time_elapsed:.4f}")
#
#             # times_taken_avg[i] += time_elapsed
#
#         times_taken_avg[i] = np.mean(curr_times_taken)
#         tasks_completed_avg[i] = np.mean(curr_tasks_completed)
#         times_taken_var[i] = np.var(curr_times_taken)
#         tasks_completed_var[i] = np.var(curr_tasks_completed)
#         print(f"\ttotal time_elapsed {sum(curr_times_taken): .4f}")
#     df_dict = {
#         "no_agents": no_agents_arr,
#         "time_taken_avg": times_taken_avg,
#         "time_taken_var": times_taken_var,
#         "tasks_completed_avg": tasks_completed_avg,
#         "tasks_completed_var": tasks_completed_var
#     }
#     df = pd.DataFrame.from_dict(df_dict)
#     df.to_csv("benchmarks/no_agents_vs_t_tc_rand.csv", index=False)
#
#
# def benchmark_tp_no_timesteps():
#     no_agents = 5
#     # max_t = 500
#     no_timesteps_arr = [100, 200, 300, 400, 500]
#     n = len(no_timesteps_arr)
#     no_runs = 20
#
#     print("##############")
#     print("REAL WAREHOUSE")
#     print("##############")
#
#     # Real Warehouse Layout
#     grid = Warehouse.txt_to_grid("map_warehouse_1.txt", use_curr_workspace=True, simple_layout=False)
#     y_len = len(grid)
#     non_task_endpoints = [(0, y) for y in range(y_len)]
#
#     # Time Taken Vs No of Agents
#
#     times_taken_avg = [0.0] * n
#     tasks_completed_avg = [0.0] * n
#     times_taken_var = [0.0] * n
#     tasks_completed_var = [0.0] * n
#
#     for i, no_timesteps in enumerate(no_timesteps_arr):
#         print(f"\nno_timesteps = {no_timesteps}")
#         start_locs = non_task_endpoints[:no_agents]
#         curr_tasks_completed = []
#         curr_times_taken = []
#
#         for run_no in range(no_runs):
#             print(f"\trun_no={run_no} - ", end="")
#             start = time.time()
#             tp = TokenPassing(grid, no_agents, start_locs, non_task_endpoints, no_timesteps, task_frequency=1,
#                               is_logging_collisions=False)
#             final_agents = tp.compute()
#             time_elapsed = time.time() - start
#             curr_tasks_completed.append(tp.get_no_tasks_completed())
#             curr_times_taken.append(time_elapsed)
#             print(f"time_elapsed: {time_elapsed:.4f}")
#
#             # times_taken_avg[i] += time_elapsed
#
#         times_taken_avg[i] = np.mean(curr_times_taken)
#         tasks_completed_avg[i] = np.mean(curr_tasks_completed)
#         times_taken_var[i] = np.var(curr_times_taken)
#         tasks_completed_var[i] = np.var(curr_tasks_completed)
#         print(f"\ttotal time_elapsed {sum(curr_times_taken): .4f}")
#     df_dict = {
#         "no_timesteps": no_timesteps_arr,
#         "time_taken_avg": times_taken_avg,
#         "time_taken_var": times_taken_var,
#         "tasks_completed_avg": tasks_completed_avg,
#         "tasks_completed_var": tasks_completed_var
#     }
#     df = pd.DataFrame.from_dict(df_dict)
#     df.to_csv("benchmarks/no_timesteps_vs_t_tc_real.csv", index=False)
#
#     print("##############")
#     print("RAND WAREHOUSE")
#     print("##############")
#     num_storage_locs = 560  # 560
#     storage_shape = (22, 44)
#
#     # Time Taken Vs No of Agents
#
#     times_taken_avg = [0.0] * n
#     tasks_completed_avg = [0.0] * n
#     times_taken_var = [0.0] * n
#     tasks_completed_var = [0.0] * n
#
#     for i, no_timesteps in enumerate(no_timesteps_arr):
#         print(f"\nno_timesteps = {no_timesteps}")
#         curr_tasks_completed = []
#         curr_times_taken = []
#
#         for run_no in range(no_runs):
#             print(f"\trun_no={run_no} - generating grid... ", end="")
#             grid = Warehouse.get_uniform_random_grid(storage_shape, num_storage_locs)
#             y_len = len(grid)
#             non_task_endpoints = [(0, y) for y in range(y_len)]
#             start_locs = non_task_endpoints[:no_agents]
#
#             print(f"doing MAPD... ", end="")
#             start = time.time()
#             tp = TokenPassing(grid, no_agents, start_locs, non_task_endpoints, no_timesteps, task_frequency=1,
#                               is_logging_collisions=False)
#             final_agents = tp.compute()
#             time_elapsed = time.time() - start
#             curr_tasks_completed.append(tp.get_no_tasks_completed())
#             curr_times_taken.append(time_elapsed)
#             print(f"time_elapsed: {time_elapsed:.4f}")
#
#             # times_taken_avg[i] += time_elapsed
#
#         times_taken_avg[i] = np.mean(curr_times_taken)
#         tasks_completed_avg[i] = np.mean(curr_tasks_completed)
#         times_taken_var[i] = np.var(curr_times_taken)
#         tasks_completed_var[i] = np.var(curr_tasks_completed)
#         print(f"\ttotal time_elapsed {sum(curr_times_taken): .4f}")
#     df_dict = {
#         "no_timesteps": no_timesteps_arr,
#         "time_taken_avg": times_taken_avg,
#         "time_taken_var": times_taken_var,
#         "tasks_completed_avg": tasks_completed_avg,
#         "tasks_completed_var": tasks_completed_var
#     }
#     df = pd.DataFrame.from_dict(df_dict)
#     df.to_csv("benchmarks/no_timesteps_vs_t_tc_rand.csv", index=False)
#
#
# def benchmark_tp():
#     # benchmark_tp_no_agents()
#     benchmark_tp_no_timesteps()
#
#
# def benchmark_warehouse():
#     no_agents = 5
#     no_timesteps = 250
#     task_frequency = 1
#     no_iters = 50
#
#     grid = Warehouse.txt_to_grid("map_warehouse_1.txt", use_curr_workspace=True, simple_layout=False)
#     y_len = len(grid)
#     x_len = len(grid[0])
#
#     non_task_endpoints = [(y, 0) for y in range(y_len)]
#     start_locs = non_task_endpoints[:no_agents]
#
#     t_elap_arr = []
#     t_elap_c_astar_arr = []
#     c_astar_calls_arr = []
#     tasks_completed_arr = []
#
#     for i in range(no_iters):
#         tp = TokenPassing(grid, no_agents, start_locs, non_task_endpoints, no_timesteps, task_frequency=task_frequency,
#                           is_logging_collisions=False)
#         print(f"Iteration {i+1}/{no_iters}...")
#         tp.compute()
#         t_elap = tp.time_elapsed
#         t_elap_c_astar = tp.time_elapsed_c_astar
#         c_astar_calls = tp.coop_astar_calls
#         tasks_completed = tp.get_no_tasks_completed()
#         print(f"\tTotal Time Elapsed: {t_elap:.4f}s")
#         print(f"\tCoop AStar Time Elapsed: {t_elap_c_astar:.4f}s ({t_elap_c_astar/t_elap*100:.2f}% of total)")
#         print(f"\tCoop AStar Calls: {c_astar_calls}")
#         print(f"\tNo of Tasks Completed: {tasks_completed}")
#         t_elap_arr.append(t_elap)
#         t_elap_c_astar_arr.append(t_elap_c_astar)
#         c_astar_calls_arr.append(c_astar_calls)
#         tasks_completed_arr.append(tasks_completed)
#
#     df_dict = {
#         "time_elap": t_elap_arr,
#         "time_elap_c_astar": t_elap_c_astar_arr,
#         "c_astar_calls": c_astar_calls_arr,
#         "tasks_completed": tasks_completed_arr
#     }
#     df = pd.DataFrame.from_dict(df_dict)
#
#     df.to_csv(f"benchmarks/na_{no_agents}_nt_{no_timesteps}_tf_{task_frequency}.csv")


def main():
    num_storage_locs = 560  # 560
    # ufg = UniformRandomGrid()
    # grid = ufg.get_uniform_random_grid()
    real_warehouse = RealWarehouse()
    grid = real_warehouse.create_grid()

    # grid = Warehouse.txt_to_grid("map_warehouse_1.txt", use_curr_workspace=True, simple_layout=False)
    y_len = len(grid)
    x_len = len(grid[0])

    no_agents = 60
    max_t = 600

    non_task_endpoints = real_warehouse.non_task_endpoints
    non_task_endpoints = [(x, y) for (y, x) in non_task_endpoints]
    start_locs = non_task_endpoints[:no_agents]

    # non_task_endpoints = [(0, y) for y in range(y_len)]
    # start_locs = non_task_endpoints[:no_agents]

    tp = TokenPassing(grid, no_agents, start_locs, non_task_endpoints, max_t, task_frequency=1,
                      is_logging_collisions=True)
    final_agents = tp.compute()

    # new_vis.animate_multi_path(full_paths, is_pos_xy=False)
    # new_vis.animate_path(full_paths_dict[0], is_pos_xy=False)
    # new_vis.window.getMouse()

    # no_tasks_completed = sum([agent.get_no_tasks_completed() for agent in final_agents])

    print(f"\nNumber of tasks completed: {tp.get_no_tasks_completed()}\n")
    print(f"Time Elapsed: {tp.time_elapsed:.4f}")
    print(f"Coop AStar Elapsed: {tp.time_elapsed_c_astar:.4f} ({tp.time_elapsed_c_astar/tp.time_elapsed * 100:.2f}% of total)")
    print(f"Coop AStar Calls: {tp.coop_astar_calls}")

    for agent in final_agents:
        print(f"############")
        print(f"############")
        print(f"# AGENT {agent.id:2} #")
        print(f"############")
        print(f"Tasks Completed: {agent.get_no_tasks_completed()}")
        print(f"Path History: {agent.path_history}")
        print(f"Task History: {agent.task_history}")
        print(f"Current Task: {agent._curr_task}")
        print(f"Current Path: {agent._curr_path}")
    # visualise_paths(grid, final_agents)
    # plot_graph(tp._graph, "tp_G.png")

    vis = VisGrid(grid, (900, 400), 25, tick_time=0.2)
    vis.window.getMouse()
    vis.animate_mapd(final_agents, is_pos_xy=True)
    vis.window.getMouse()


if __name__ == "__main__":
    from Visualisation.vis import VisGrid
    main()
    # benchmark_tp()
    # benchmark_warehouse()
