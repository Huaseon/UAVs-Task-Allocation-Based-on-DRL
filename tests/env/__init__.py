"""
动态任务分配 & 多目标优化 & 异构无人机 & 多任务分配
"""
# %%
from .. import cur_path, SEED
cur_path = cur_path / "env"

from .data.d_drone import DRONE_FEATURES_SIZE
from .data.d_task import TASK_FEATURES_SIZE

# %%
AVG_DRONE_COUNT = 5
MIN_DRONE_COUNT = 3
MAX_DRONE_COUNT= AVG_DRONE_COUNT * 2 - MIN_DRONE_COUNT

AVG_TASK_COUNT_PER_DRONE = 5
DESCRIPTION = "动态任务分配 & 多目标优化 & 异构无人机 & 多任务分配"
