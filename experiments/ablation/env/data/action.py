# %%
from . import SEED
from .d_drone import DroneSpace
from .d_task import TaskSpace
from .utils import step

# %%
import numpy as np
np.random.seed(SEED)

# %%
class ActionSpace:
    def __init__(self):
        """
        `_eye`：参与分配的无人机数目的规格的单位矩阵
        `state`：决策方案
        `info`：无人机数目 * 任务数目 的 0-1矩阵
        """
        self._eye = None
        self.state = None
        self.info = None
    
    def _reset(self, drone_size: int, task_size: int) -> np.ndarray:
        """
        重置动作空间参数
        ``
        `drone_size`：无人机数目
        `task_size`：任务数目
        返回决策记录，应为空
        """
        self._eye = np.eye(drone_size)
        self.state = np.array([], dtype=np.int64)
        self.info = np.zeros(shape=(task_size, drone_size))

        assert np.array_equal(self.info.argmax(axis=1)[self.info.any(axis=1)], self.state)

        return self.state
    
    def step(self, drone_space: DroneSpace, task_space: TaskSpace, action: np.ndarray) -> np.ndarray:
        action = action.astype(np.int64)
        assert np.array_equal(self.info.argmax(axis=1)[self.info.any(axis=1)], action[:-task_space.cur_tasks_count])
        self.info[len(self.state):len(self.state) + task_space.cur_tasks_count] = self._eye[action[-task_space.cur_tasks_count:]]
        self.state = action
        task_space.cur_tasks_count = np.random.randint(low=task_space.tasks.size > len(action), high=(task_space.tasks.size - len(action) + 1)//2 + 1)

        drones= drone_space.drones
        tasks = task_space.tasks[:len(action)]
        rewards = []

        for idx, drone in enumerate(drones):
            _tasks = tasks[action==idx]
            reward = step(drone=drone, tasks=_tasks)
            rewards.append(reward)

        return np.array(rewards)
    
    def done(self) -> bool:
        return self.info.any(axis=1).all()

    def obs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        局部观测
        返回决策方案以及 0-1 矩阵
        """
        assert np.array_equal(self.info.argmax(axis=1)[self.info.any(axis=1)], self.state)

        return self.state, self.info

    def _obs(self) -> np.ndarray:
        """
        观测全局决策方案
        返回决策方案
        """
        assert np.array_equal(self.info.argmax(axis=1)[self.info.any(axis=1)], self.state)
       
        return self.state

    def _info(self) -> np.ndarray:
        """
        观测全局决策信息
        返回 0-1 矩阵
        """
        assert np.array_equal(self.info.argmax(axis=1)[self.info.any(axis=1)], self.state)

        return self.info



