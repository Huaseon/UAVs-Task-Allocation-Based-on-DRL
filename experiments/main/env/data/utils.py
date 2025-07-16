# %%
from . import SEED
from .d_drone import Drone
from .d_task import Task

# %%
import numpy as np
np.random.seed(SEED)

# %%
def step(drone: Drone, tasks: np.ndarray[Task]) -> float:
    pld_limit = payload_limit(drone=drone, tasks=tasks) # [0, 1]
    if pld_limit < 1 + 1/3:
        return 0
    
    tpr_limit = temporal_limit(drone=drone, tasks=tasks) # [0, 1]
    if tpr_limit < 1/3:
        return 0
    
    opm_reward = optimization_reward(drone=drone, tasks=tasks) # [0, inf]
    return opm_reward

# %%
def payload_limit(drone: Drone, tasks: np.ndarray[Task]) -> float:
    drone_payload = drone.state[-1]
    tasks_payload = np.array([task.state[-2] * (1 + (1 - satisfy(drone=drone, task=task)) * 1/3)  for task in tasks]).sum() if tasks.size else 0
    return np.min(1, drone_payload / (tasks_payload + 1e-9))

# %%
def temporal_limit(drone: Drone, tasks: np.ndarray[Task]) -> float:
    tasks_last_lifetime = np.array([task.state[-1] * (1 - (1 - satisfy(drone=drone, task=task)) * 1/3) for task in tasks])
    tasks_last_lifetime.sort()
    gamma = tasks_last_lifetime / (tasks_last_lifetime.cumsum() + 1e-9)
    return gamma.min()

# %%
def optimization_reward(drone: Drone, tasks: np.ndarray[Task]) -> float:
    """
    多目标优化：载荷 + 时间
    """
    pyl_reward = payload_reward(drone=drone, tasks=tasks)
    tmp_reward = temporal_reward(drone=drone, tasks=tasks)
    return pyl_reward * tmp_reward

# %%
def payload_reward(drone: Drone, tasks: np.ndarray[Task]) -> float:
    drone_values = np.array(drone.state[-1:-4:-1].tolist())
    tasks_values = np.array([task.state[-2:-5:-1].tolist() for task in tasks])
    tasks_values[:, 0] *= 1 + (1 - np.array([satisfy(drone=drone, task=task) for task in tasks])) * 1 / 3

    cur_drone = np.array(drone_values.tolist())
    cur_tasks = np.array(tasks_values[tasks_values[:, 0] <= cur_drone[0]].tolist())
    
    payloads = 0
    while cur_tasks.size > 0:
        drone_point = cur_drone[[1, 2]]
        tasks_point = cur_tasks[:, [1, 2]]
        d = get_distances(drone_point=drone_point, tasks_point=tasks_point)
        indice = (cur_tasks[:, 0] / (d + 1e-9)).argmax()
        selected, cur_tasks = cur_tasks[indice], np.vstack((cur_tasks[:indice], cur_tasks[indice+1:]))

        payload_delta = selected[0]
        cur_drone[0], cur_drone[1:], payloads = cur_drone[0] - payload_delta, selected[1:], payloads + payload_delta

        cur_tasks = cur_tasks[cur_tasks[:, 0] <= cur_drone[0]]

    return payloads

# %%
def temporal_reward(drone: Drone, tasks: np.ndarray[Task]) -> float:

    tasks_type = np.vstack([task._info().get("TASK-TYPE") for task in tasks])
    drone_values = np.array([0] + [0] + drone.state[3:5].tolist())
    tasks_values = np.array([[id] + task.state[-1:].tolist() + task.state[3:5].tolist() for id, task in enumerate(tasks)])
    tasks_values[:, 1] *= 1 - (1 - np.array([satisfy(drone=drone, task=task) for task in tasks])) * 1 / 3

    # [id, tpr, x, y]
    cur_drone = np.array(drone_values.tolist())
    cur_tasks = np.array(tasks_values[tasks_values[:, 1] > 0].tolist())

    times = 0
    while cur_tasks.size > 0:
        drone_point = cur_drone[[2, 3]]
        tasks_point = cur_tasks[:, [2, 3]]
        d = get_distances(drone_point=drone_point, tasks_point=tasks_point)
        indice = (cur_tasks[:, 1] / (d + 1e-9)).argmax()
        selected, cur_tasks = cur_tasks[indice], np.vstack((cur_tasks[:indice], cur_tasks[indice+1:]))
        if np.array_equal(tasks_type[selected[0].astype(int)], [1, 0, 0]):
            """
            跟踪
            """
            time_delta = 2 * selected[1] / 3
            cur_drone, cur_tasks[:, 1], times = selected, cur_tasks[:, 1] - time_delta, times + time_delta
        
        elif np.array_equal(tasks_type[selected[0].astype(int)], [0, 1, 0]):
            """
            物流
            """
            time_delta = selected[1] * 1 / 4
            cur_drone, cur_tasks[:, 1], times = selected, cur_tasks[:, 1] - time_delta, times + selected[1] - time_delta
        
        elif np.array_equal(tasks_type[selected[0].astype(int)], [0, 0, 1]):
            """
            巡检
            """
            cur_tasks = np.vstack((cur_drone, cur_tasks))
            time_delta = selected[1] *  1 / 3
            cur_drone, cur_tasks[:, 1], times = selected, cur_tasks[:, 1] - time_delta, times + time_delta
            cur_drone[1] = cur_drone[1] - time_delta
        
        else:
            raise ValueError(f"Wrong the task_dtype: {tasks_type[selected[0].astype(int)]}")

        cur_tasks = cur_tasks[cur_tasks[:, 1] > 0]

    return times

# %%
def get_distances(drone_point: np.ndarray, tasks_point: np.ndarray) -> np.ndarray:
    return np.sqrt(((tasks_point - drone_point) ** 2).sum(axis=1) + 1e-9)

def satisfy(drone: Drone, task: Task) -> bool:
    return (drone.state[:3] & task.state[:3]).any()
