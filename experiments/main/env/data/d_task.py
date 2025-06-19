# %%
from . import SEED

# %%
import json
from pathlib import Path

# %%
import numpy as np
np.random.seed(SEED)

# %%
from . import cur_path
_file_path: Path = cur_path / "d_task.json"

# %%
from .d_utils import flatten_dict, value_sample, state2info

# %%
_task_detail = None

with open(_file_path, 'r') as f:
    _task_detail: dict = json.load(f)

_flatten_info: dict = flatten_dict(input_dict=_task_detail)

_sample_state = np.hstack([value_sample(value=value, size=size) for value, size in _flatten_info.values()])
TASK_FEATURES_SIZE = _sample_state.size

# %%
class Task:
    
    detail = _task_detail
    info = _flatten_info
    dim = TASK_FEATURES_SIZE

    def __init__(self):
        self.state = None
        self._reset()

    def _reset(self) -> None:
        self.state = np.hstack([value_sample(value=value, size=size) for value, size in self.info.values()])
    
    def _obs(self) -> np.ndarray:
        return self.state
    
    def _info(self) -> dict:
        return state2info(state=self.state, info_template=self.info)


# %%
class TaskSpace:

    """
    `动态任务` & `多任务分配`
    """

    detail = _task_detail
    info = _flatten_info
    dim = TASK_FEATURES_SIZE

    def __init__(self):
        """
        `count`：任务数目
        `tasks`：任务对象列表
        `state`：任务状态矩阵
        `info`：任务信息字典列表
        `cur_tasks_count`：当前时间步新增任务数目
        """
        self.count = 0
        self.tasks = None
        self.state = None
        self.info = None
        self.cur_tasks_count = None
    
    def _reset(self, size: int) -> np.ndarray:
        """
        重置空间参数
        `size`：任务数目
        生成 `size` 数目任务，记录任务状态矩阵和信息字典列表
        返回 `任务状态矩阵`
        """
        self.count = size
        self.tasks = np.array([Task() for _ in range(self.count)])
        self.state = np.vstack([task._obs() for task in self.tasks])
        self.info = [task._info() for task in self.tasks]
        self.cur_tasks_count = np.random.randint(low=1, high=(size + 1) // 2 + 1)
        return self.state
    
    def obs(self, mask: np.ndarray) -> np.ndarray:
        """
        局部观测
        `mask`：`task_size` * `drone_size` 的 0-1 矩阵
        返回 `已分配任务状态` + `未分配的前m个任务状态`
        """
        done = mask.any(axis=1)
        remainder = ~done
        return np.vstack((self.state[done], self.state[remainder][:self.cur_tasks_count]))

    def _obs(self) -> np.ndarray:
        """
        观测全局任务状态
        返回 `任务状态矩阵`
        """
        return self.state

    def _info(self) -> list[dict]:
        """
        观测全局任务信息
        返回 `任务信息字典列表`
        """
        return self.info
    

        
