# %5
from . import SEED

# %%
import numpy as np

# %%
np.random.seed(SEED)

# %%
def to_reward(rewards: np.ndarray) -> tuple[np.ndarray, float]:
    return np.mean(rewards[:, :-1], axis=0), np.linalg.norm(rewards[:, -1])