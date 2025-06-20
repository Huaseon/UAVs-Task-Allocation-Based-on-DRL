from . import SEED
# %%
import numpy as np
np.random.seed(SEED)

def avg_length(avg_tasks_per: int):
    a = np.zeros(shape=(avg_tasks_per + 1,))
    for i in range(1, avg_tasks_per + 1):
        a[i] = 1 + a[i-(i+1)//2:i].mean()
    return a[-1]