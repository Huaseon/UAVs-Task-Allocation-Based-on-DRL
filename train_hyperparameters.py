SEED = 20040508

import numpy as np
from pathlib import Path
from hashlib import sha256

from ray import tune, train
from ray.tune import Tuner, RunConfig
from ray.tune.schedulers import ASHAScheduler

from experiments.hyperparameters.agent import Agent
from experiments.hyperparameters.env.simulator import Environment
from experiments.hyperparameters.env import AVG_DRONE_COUNT, AVG_TASK_COUNT_PER_DRONE

from tests.agent import Agent as TestAgent
from tests.env.simulator import Environment as TestEnvironment

from utils.tools import avg_length

np.random.seed(SEED)

length_per_episode = np.floor(avg_length(avg_tasks_per=AVG_TASK_COUNT_PER_DRONE * AVG_DRONE_COUNT)).astype(int)

save_dir = Path(".") / "output"/ "hyperparameters"
save_dir.exists() or save_dir.mkdir(parents=True)

def train_one_episode(agent: Agent):
    env = Environment()
    drone_state, task_state, action_state = env.reset()
    while True:
        action = agent.act(drone_state=drone_state, task_state=task_state, action_state=action_state)
        next_drone_state, next_task_state, next_action_state, reward, done = env.step(action=action)
        agent.cache(drone_state=drone_state, task_state=task_state, action_state=action_state, next_drone_state=next_drone_state, next_task_state=next_task_state, action=action, reward=reward, done=done)
        q, loss = agent.learn()
        if done:
            break
        drone_state, task_state, action_state = next_drone_state, next_task_state, next_action_state
    return q, loss

def test_episodes(save_model):
    env = TestEnvironment()
    test_agent = TestAgent.load(save_model=save_model)
    total_reward = []
    total_q = []
    total_r = []
    episodes = 100
    for episode in range(episodes):
        drone_state, task_state, action_state = env.reset()
        assert action_state.size == 0
        while True:
            q, action = test_agent.act(drone_state=drone_state, task_state=task_state, action_state=action_state)
            assert np.array_equal(action[:len(action_state)], action_state)
            next_drone_state, next_task_state, next_action_state, reward, done, r = env.step(action=action)
            assert np.array_equal(next_task_state[:len(task_state)], task_state)
            assert np.array_equal(next_action_state, action)
            drone_state, task_state, action_state = next_drone_state, next_task_state, next_action_state
            total_reward.append(reward)
            total_q.append(q)
            if done:
                break
        total_r.append(r)
    avg_reward = np.round(np.sum(total_reward) / episodes, 3)
    avg_q = np.round(total_q, 3)
    avg_r = np.mean(total_r, axis=0)
    return avg_reward, avg_q, avg_r

def train_agent(config):
    lr, sync_every_rate, gamma, max_times, exploration_rate_min = config['lr'], config['sync_every_rate'], config['gamma'], config['max_times'], config['exploration_rate_min']
    hp = f"{lr}-{sync_every_rate}-{gamma}-{max_times}-{exploration_rate_min}"
    name = sha256(hp.encode()).hexdigest()[::8]
    agent = Agent(save_dir=save_dir / name, lr=lr, sync_every_rate=sync_every_rate, gamma=gamma, max_times=max_times, exploration_rate_min=exploration_rate_min)

    total_q, total_loss = [], []
    for episode in range(config["episodes_per_tial"]):
        episode_q, episode_loss = train_one_episode(agent=agent)
        total_q.append(episode_q)
        total_loss.append(episode_loss)
        if (episode + 1) % 100 == 0:
            save_model = dict(
                model=agent.net.state_dict(),
                is_base=agent.is_base,
                gamma=agent.gamma
            )
            avg_reward, avg_q, avg_r = test_episodes(save_model=save_model)
            train.report(
                metrics={
                    "mean_reward": avg_reward,
                    "mean_q": avg_q,
                    "mean_r": avg_r
                }
            )
    save_model = dict(
        model=agent.net.state_dict(),
        is_base=agent.is_base,
        gamma=agent.gamma
    )
    avg_reward, avg_q, avg_r = test_episodes(save_model=save_model)
    train.report(mean_reward=avg_reward, mean_q=avg_q, mean_r=avg_r)
    
    return {"mean_reward": avg_reward, "mean_q": avg_q, "mean_r": avg_r}

def tune_hp():
    config = {
        "lr": tune.loguniform(1e-6, 1e-4),
        "sync_every_rate": tune.uniform(5e-3, 5e-2),
        "gamma": tune.uniform(0.5, 0.999),
        "max_times": 1,
        "exploration_rate_min": 1,
        "episodes_per_tial": 6000
    }

    scheduler = ASHAScheduler(
        metric="mean_reward",
        mode="max",
        max_t=60,
        grace_period=3,
        reduction_factor=3
    )

    storage_path = save_dir.resolve()
    
    run_config = RunConfig(
        storage_path=storage_path,
        name="hp_agent"
    )

    trainable = tune.with_resources(
        train_agent,
        resources={"cpu": 1, "gpu": .3}
    )

    tuner = Tuner(
        trainable=trainable,
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=50,
            scheduler=scheduler,
        ),
        run_config=run_config
    )

    results = tuner.fit()

    return results

if __name__ == "__main__":
    bestconfig = tune_hp()

