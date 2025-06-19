SEED = 20040508

from experiments.main.env.simulator import Environment
import numpy as np
np.random.seed(SEED)

env = Environment()

from experiments.main.agent import Agent
from pathlib import Path
from utils.log import AgentLogger

max_times = 4
exploration_rate_min = 3/4

from experiments.main.env import AVG_DRONE_COUNT, AVG_TASK_COUNT_PER_DRONE
from utils.tools import avg_length
top = (10000 * max_times // np.floor(avg_length(avg_tasks_per=AVG_TASK_COUNT_PER_DRONE * AVG_DRONE_COUNT)).astype(int) + 1)

save_dir = Path("output") / "main" / f"{max_times}-{exploration_rate_min:.4f}" / "base"
save_dir.mkdir(parents=True) if not save_dir.exists() else None

base_agent = Agent(save_dir=save_dir, is_base=True, max_times=max_times, exploration_rate_min=exploration_rate_min)
logger = AgentLogger(save_dir=save_dir, name="base", top=top)

for episode in range(top):
    drone_state, task_state, action_state = env.reset()

    while True:
        action = base_agent.act(drone_state=drone_state, task_state=task_state, action_state=action_state)

        next_drone_state, next_task_state, next_action_state, reward, done = env.step(action=action)

        base_agent.cache(drone_state=drone_state, task_state=task_state, action_state=action_state, next_drone_state=next_drone_state, next_task_state=next_task_state, action=action, reward=reward, done=done)
        
        q, loss = base_agent.learn()
        
        logger.log_step(reward=reward, loss=loss, q=q)

        drone_state, task_state, action_state = next_drone_state, next_task_state, next_action_state

        if done:
            break
    
    logger.log_episode()

    logger.record(episode=episode, epsilon=base_agent.exploration_rate, step=base_agent.curr_step)
    logger.wind.save([logger.name])

base_agent.save()
logger.wind.save([logger.name])