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
_file_path: Path = cur_path / "d_drone.json"

# %%
from .d_utils import flatten_dict, value_sample, state2info

# %%
_drone_detail = None

with open(_file_path, 'r') as f:
    _drone_detail: dict = json.load(f)

_flatten_info: dict = flatten_dict(input_dict=_drone_detail)

_sample_state = np.hstack([value_sample(value=value, size=size) for value, size in _flatten_info.values()])

DRONE_FEATURES_SIZE = _sample_state.size

# %%
class Drone:
    
    detail = _drone_detail
    info = _flatten_info
    dim = DRONE_FEATURES_SIZE

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
class DroneSpace:

    """
    `动态无人机` & `异构无人机`
    """

    detail = _drone_detail
    info = _flatten_info
    dim = DRONE_FEATURES_SIZE

    def __init__(self):
        """
        `count`: 无人机数目
        `drones`: 无人机对象列表
        `state`: 无人机状态矩阵
        `info`: 无人机信息字典列表
        """
        self.count = 0
        self.drones = None
        self.state = None
        self.info = None
    
    def _reset(self, size: int) -> np.ndarray:
        """
        重置空间参数
        `size`：无人机数目
        生成 `size` 数目无人机，记录无人机状态矩阵与信息字典列表
        返回 `无人机状态矩阵`
        """
        self.count = size
        self.drones = np.array([Drone() for _ in range(self.count)])
        self.state = np.vstack([drone._obs() for drone in self.drones])
        self.info = [drone._info() for drone in self.drones]
        return self.state
    
    def obs(self) -> np.ndarray:
        """
        局部观测
        返回 `无人机状态信息`
        """
        return self.state

    def _obs(self) -> np.ndarray:
        """
        观测全局无人机状态
        返回 `无人机状态矩阵`
        """
        return self.state

    def _info(self) -> list[dict]:
        """
        观测全局无人机信息
        返回 `无人机信息字典列表`
        """
        return self.info
    

        
