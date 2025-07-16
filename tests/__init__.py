# %%
from pathlib import Path

# %%
cur_path = Path(".") / "tests"
SEED = 31415926

"""
hyperparameters: {
    "lr": "学习率",
    "sync_every_rate": "参数同步频率",
    "gamma": "折扣因子"
}
"""

HYPERPARAMETERS = {
    "lr": 1e-5,
    "sync_every_rate": 1e-2,
    "gamma": .8
}
