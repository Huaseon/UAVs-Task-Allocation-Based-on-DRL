# %%
from .. import cur_path, SEED

# %%
cur_path = cur_path / "main"

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

