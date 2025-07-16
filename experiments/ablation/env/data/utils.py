# %%
from . import SEED
from .d_drone import Drone
from .d_task import Task

# %%
import numpy as np
np.random.seed(SEED)

# %%
def step(drone: Drone, tasks: np.ndarray[Task]) -> float:
    beta = tasks.size
    if beta == 0: return 0

    tasks = legal_tasks(drone=drone, tasks=tasks)
    
    beta = tasks.size / beta
    if beta == 0: return 0

    pyl_limit = payload_limit(drone=drone, tasks=tasks) # [0, 1]
    tpr_limit = temporal_limit(drone=drone, tasks=tasks) # [0, 1]
    opm_reward = optimization_reward(drone=drone, tasks=tasks)

    return beta * pyl_limit * tpr_limit * opm_reward

# %%
def payload_limit(drone: Drone, tasks: np.ndarray[Task]) -> float:
    drone_payload = drone.state[-1]
    tasks_payload = np.array([task.state[-2] for task in tasks]).sum()
    return np.sqrt(drone_payload >= tasks_payload or max(1 - (tasks_payload - drone_payload) / (drone_payload + 1e-9), 0))

# %%
def temporal_limit(drone: Drone, tasks: np.ndarray[Task]) -> float:
    tasks_last_lifetime = np.array([task.state[-1] for task in tasks])
    tasks_last_lifetime.sort()
    gamma = tasks_last_lifetime / (tasks_last_lifetime.cumsum() + 1e-9)
    return np.sqrt(gamma.min())

# %%
def optimization_reward(drone: Drone, tasks: np.ndarray[Task]) -> float:
    """
    多目标优化：载荷 + 时间
    """
    pyl_reward = payload_reward(drone=drone, tasks=tasks)
    tmp_reward = temporal_reward(drone=drone, tasks=tasks)

    return pyl_reward + tmp_reward

# %%
def payload_reward(drone: Drone, tasks: np.ndarray[Task]) -> float:
    drone_values = np.array(drone.state[-1:-4:-1].tolist())
    tasks_values = np.array([task.state[-2:-5:-1].tolist() for task in tasks])
    
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
    

    # [id, tpr, x, y]
    cur_drone = np.array(drone_values.tolist())
    cur_tasks = np.array(tasks_values[tasks_values[:, 1] > 0].tolist())

    times = 0
    while cur_tasks.size > 0:
        cur_drone[1:], cur_tasks[:, 1:] = dynamic_state(state=cur_drone[1:]), dynamic_state(cur_tasks[:, 1:])
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
def legal_tasks(drone: Drone, tasks: np.ndarray[Task]) -> np.ndarray[Task]:
    drone_type = drone._info().get("DRONE-TYPE").astype(int)
    tasks_type = np.vstack([task._info().get("TASK-TYPE") for task in tasks]).astype(int)
    return tasks[(tasks_type & drone_type).any(axis=1)]

# %%
def get_distances(drone_point: np.ndarray, tasks_point: np.ndarray) -> np.ndarray:
    return np.sqrt(((tasks_point - drone_point) ** 2).sum(axis=1) + 1e-9)

# %%
def dynamic_state(state: np.ndarray):
    state_delta = (np.random.random(state.shape) - 0.5) / 10
    return state * (1 + state_delta)