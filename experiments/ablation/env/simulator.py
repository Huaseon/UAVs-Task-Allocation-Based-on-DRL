
# %%
from .data.d_drone import DroneSpace
from .data.d_task import TaskSpace
from .data.action import ActionSpace
from .utils import to_reward
from . import SEED, MIN_DRONE_COUNT, MAX_DRONE_COUNT, AVG_TASK_COUNT_PER_DRONE, DESCRIPTION

# %%
import numpy as np
np.random.seed(SEED)

# %%
import random
random.seed(SEED)


# %%
class Environment:
    """
    `动态任务分配` & `多目标优化` & `异构无人机` & `多任务分配`
    """
    description = DESCRIPTION

    def __init__(self):
        """
        `drone_count`：无人机数目区间 声明无人机最小数目和最大数目
        `avg_task_count_per_drone`：平均每架无人机分配任务数目
        `drone_space`：无人机空间 由作业无人机组成
        `task_space`：任务空间 由任务目标组成
        `action_space`：动作空间 记录决策方案
        `cur_reward`：当下奖励记录
        """
        self.drone_count = (MIN_DRONE_COUNT, MAX_DRONE_COUNT)
        self.avg_task_count_per_drone = AVG_TASK_COUNT_PER_DRONE

        self.drone_space = DroneSpace()
        self.task_space = TaskSpace()
        self.action_space = ActionSpace()
        self.cur_reward = None

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        重置环境参数
        重置无人机空间：生成随机 `drone_size` 个无人机
        重置任务空间：生成 `task_size` *  `avg_task_count_per_drone` 个任务目标
        重置动作空间：初始化决策方案
        重置奖励记录

        返回 局部观测，对`无人机空间局部观测`、`任务空间局部观测`、`动作空间局部观测`
        """
        drone_size = random.randint(*self.drone_count)
        task_size = drone_size * self.avg_task_count_per_drone
        
        self.drone_space._reset(drone_size)
        self.task_space._reset(task_size)
        self.action_space._reset(drone_size, task_size)
        self.cur_reward = 0

        return self.obs()
    
    def obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        局部观测
        返回当前时刻`无人机状态`、`任务状态`、`决策方案`
        """
        drone_state = self.drone_space.obs()
        action_state, action_info = self.action_space.obs()
        task_state = self.task_space.obs(mask=action_info)
        return drone_state, task_state, action_state

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        """
        `action`：作出的决策
        决策更新 -> (状态更新，奖励)
        返回`无人机状态局部观测`，`任务状态局部观测`，`决策状态`，`奖励`，`终止信号`
        """

        rewards = self.action_space.step(drone_space=self.drone_space, task_space=self.task_space, action=action)
        reward = to_reward(rewards=rewards)
        reward_delta, self.cur_reward = reward - self.cur_reward, reward
        
        return *self.obs(), reward_delta, self.action_space.done()

    def _obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        观测环境全局状态信息
        返回 `无人机全局状态`、`任务全局状态`和`决策方案`
        """
        return self.drone_space._obs(), self.task_space._obs(), self.action_space._obs()

    def _info(self) -> tuple[list[dict], list[dict], np.ndarray]:
        """
        观测环境全局信息
        返回 `全局无人机信息`、`全局任务信息`和`决策信息`
        """
        return self.drone_space._info(), self.task_space._info(), self.action_space._info()

