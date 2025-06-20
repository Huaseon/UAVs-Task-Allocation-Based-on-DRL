SEED = 20040508

from tests.env.simulator import Environment
import numpy as np
np.random.seed(SEED)

env = Environment()

from tests.agent import Agent
from pathlib import Path
from utils.log import TestLogger

import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

max_times = 4
exploration_rate_min = 3/4

from_path = Path("output") / "main" / f"{max_times}-{exploration_rate_min:.4f}" / "full"
to_path = Path("output") / "test" / f"{max_times}-{exploration_rate_min:.4f}" / "full"
to_path.mkdir(parents=True) if not to_path.exists() else None

agent: Agent = Agent.loads(from_path=from_path)
assert agent

top = 1000
logger = TestLogger(save_dir=to_path, top=top, name="test-full")

for episode in range(top):
    drone_state, task_state, action_state = env.reset()
    assert action_state.size == 0
    while True:
        q, action = agent.act(drone_state=drone_state, task_state=task_state, action_state=action_state)
        assert np.array_equal(action[:len(action_state)], action_state)
        
        next_drone_state, next_task_state, next_action_state, reward, done, r = env.step(action=action)
        assert np.array_equal(next_action_state, action)
        assert np.array_equal(next_task_state[:len(task_state)], task_state)

        logger.log_step(reward=reward, q=q, r=r)

        drone_state, task_state, action_state = next_drone_state, next_task_state, next_action_state

        if done:
            break
    
    print(action)
    logger.log_episode()

    logger.record(episode=episode)
    logger.wind.save([logger.name])

logger.wind.save([logger.name])