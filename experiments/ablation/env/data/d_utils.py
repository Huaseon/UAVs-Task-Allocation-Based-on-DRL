# %%
from . import SEED

# %%
import random
random.seed(SEED)

# %%
import numpy as np
np.random.seed(SEED)

# %%
def flatten_dict(input_dict: dict, parent_key: str='', sep: str='-') -> dict:
    items = {}
    current_key = input_dict['key']
    new_parent_key = f"{parent_key}{sep}{current_key}" if parent_key else current_key

    if isinstance(input_dict['value'], list):
        for item in input_dict['value']:
            items.update(flatten_dict(item, new_parent_key, sep))
    
    else:
        items[new_parent_key] = [input_dict['value'], input_dict['size']]
    
    return items

# %%
def value_sample(value: str, size: list) -> list:
    match value, size:
        case "B-STRING", [flag, [TYPECOUNT]]:
            if flag:
                selected = random.randint(1, 2 ** TYPECOUNT - 1)
                return list(map(int, f"{bin(selected)[2:]:>03}"))
            selected = random.randint(1, TYPECOUNT)
            return [0] * (selected - 1) + [1] + [0] * (TYPECOUNT - selected)
        case "FLOAT", [MIN, MAX, [1]]:
            return [random.random() * (MAX - MIN) + MIN]
        case _:
            pass

# %%
def state2info(state: np.ndarray, info_template: dict) -> dict[str, np.ndarray]:
    cur_state = state.copy()
    info = {}
    for key, value in info_template.items():
        _, [*_, [size]] = value
        info[key], cur_state = cur_state[:size], cur_state[size:]
    return info
