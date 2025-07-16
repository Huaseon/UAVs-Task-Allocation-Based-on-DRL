# %%
from .. import cur_path, SEED

# %%
cur_path = cur_path / "hyperparameters"

HYPERPARAMETERS = {
    "lr": 1e-5,
    "sync_every_rate": 1e-2,
    "gamma": .8
}