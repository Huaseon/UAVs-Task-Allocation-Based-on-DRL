# %5
from . import SEED

# %%
import numpy as np

# %%
np.random.seed(SEED)

# %%
def to_reward(rewards: np.ndarray) -> float:
    return np.linalg.norm(rewards)
