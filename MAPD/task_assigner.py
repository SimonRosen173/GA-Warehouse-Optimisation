from typing import List, Dict, Set, Optional, Tuple
from numpy import random


class Task:
    def __init__(self, id, pickup_point, dropoff_point, timestep_created):
        self.id = id
        self.pickup_point: Tuple[int, int] = pickup_point
        self.dropoff_point: Tuple[int, int] = dropoff_point
        self.timestep_created = timestep_created
        self.timestep_assigned = -1
        self.timestep_completed = -1

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return f"TASK(id={self.id}) - pickup_point={self.pickup_point}, dropoff_point={self.dropoff_point}"


class TaskAssigner:
    EMPTY = 0
    PICKUP = 1
    DROPOFF = 2
    OBSTACLE = 3
    NON_TASK_ENDPOINT = 4

    def __init__(self, grid: List[List[int]], unreachable_locs: Set, task_frequency: int):
        self._grid = grid
        self._unreachable_locs = unreachable_locs
        self._task_frequency = task_frequency

        self._pickup_points = []
        self._dropoff_points = []

        self._ready_tasks = []
        self._complete_tasks = []

        self._task_id_no = 0

        self._process_input_grid()

        self._curr_t = -1
        self.inc_timestep()
        # print(f"Random seed {}")

        pass

    def _process_input_grid(self):
        grid = self._grid
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                curr_el = grid[y][x]
                if (x, y) not in self._unreachable_locs:
                    if curr_el == TaskAssigner.PICKUP:
                        self._pickup_points.append((x, y))
                    elif curr_el == TaskAssigner.DROPOFF:
                        self._dropoff_points.append((x, y))
                    elif curr_el == TaskAssigner.OBSTACLE:
                        pass

    def _create_task(self):
        if len(self._pickup_points) - 1 < 0 or len(self._dropoff_points) - 1 < 0:
            print("Available pickup points and dropoff points depleted")
            return
        if len(self._pickup_points) - 1 == 0:
            pickup_point_ind = 0
        else:
            pickup_point_ind = random.randint(0, len(self._pickup_points) - 1)

        if len(self._dropoff_points) - 1 == 0:
            dropoff_point_ind = 0
        else:
            dropoff_point_ind = random.randint(0, len(self._dropoff_points) - 1)

        curr_t = self._curr_t
        task_id = self._task_id_no
        self._task_id_no += 1

        pickup_point = self._pickup_points[pickup_point_ind]
        dropoff_point = self._dropoff_points[dropoff_point_ind]

        # del self._pickup_points[pickup_point_ind]
        # del self._dropoff_points[dropoff_point_ind]

        new_task = Task(task_id, pickup_point, dropoff_point, curr_t)

        # print(f"Task Created = {str(new_task)}")

        self._ready_tasks.append(new_task)

    def get_ready_tasks(self):
        return self._ready_tasks

    def get_ready_task(self):
        if len(self._ready_tasks) > 0:
            return self._ready_tasks[0]
        else:
            return None

    def remove_task_from_ready(self, task: Task):
        for i in range(len(self._ready_tasks)):
            if self._ready_tasks[i] == task:
                del self._ready_tasks[i]
                break

    def task_complete(self, task: Task):
        # task.timestep_completed = self._curr_t
        # self._dropoff_points.append(task.dropoff_point)
        # self._pickup_points.append(task.pickup_point)
        self._complete_tasks.append(task)

    def inc_timestep(self):
        self._curr_t += 1
        if self._curr_t % self._task_frequency == 0:
            self._create_task()

    def inc_timestep_by_n(self, n):
        for i in range(n):
            self.inc_timestep()

