SEED = 31415926

from tests.env.simulator import Environment
import numpy as np
np.random.seed(SEED)

env = Environment()

from pathlib import Path
from utils.log import TestLogger

import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

to_path = Path("output") / "test" / f"env"
to_path.mkdir(parents=True) if not to_path.exists() else None

top = 1000
logger = TestLogger(save_dir=to_path, top=top, name="test-env")

for episode in range(top):
    drone_state, task_state, action_state = env.reset()
    assert action_state.size == 0
    while True:
        action = np.hstack((action_state, np.random.randint(low=0, high=len(drone_state), size=(len(task_state) - len(action_state), ))))
        assert np.array_equal(action[:len(action_state)], action_state)
        
        next_drone_state, next_task_state, next_action_state, reward, done, r = env.step(action=action)
        assert np.array_equal(next_action_state, action)
        assert np.array_equal(next_task_state[:len(task_state)], task_state)

        logger.log_step(reward=reward, q=0, r=r)

        drone_state, task_state, action_state = next_drone_state, next_task_state, next_action_state

        if done:
            break
    
    # print(action)
    logger.log_episode()

    logger.record(episode=episode)
    logger.wind.save([logger.name])

logger.wind.save([logger.name])