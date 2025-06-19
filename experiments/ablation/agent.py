# %%
import torch
import numpy as np
from torch import nn
from pathlib import Path
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

# %%
from . import SEED, HYPERPARAMETERS
assert "lr" in HYPERPARAMETERS and "sync_every_rate" in HYPERPARAMETERS and "gamma" in HYPERPARAMETERS

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# %%
from .env import MAX_DRONE_COUNT, AVG_TASK_COUNT_PER_DRONE

MAX_TASK_COUNT = MAX_DRONE_COUNT * AVG_TASK_COUNT_PER_DRONE

# %%
from .model.net import PolicyNet

# %%
class Agent:
    def __init__(self, save_dir: Path, variant: str, max_times: float=3, exploration_rate_min: float=.5) -> None:
        
        test_rate = 1/3
        assert max_times * (1 - test_rate) >= 1/exploration_rate_min

        self.save_dir: Path = save_dir # 存储目录路径
        self.save_dir.mkdir(parents=True) if not self.save_dir.exists() else None

        self.variant = variant # 变体
        self.max_times = max_times # 数据体量

        self.device: torch.device = torch.device(torch.cuda.is_available() and "cuda" or "cpu") # 使用gpu或cpu

        self.net: PolicyNet = PolicyNet(variant=variant).float() # 策略模型
        self.net: PolicyNet = self.net.to(device=self.device) # 迁移模型至运算设备

        self.burnin = 2500 # 经验池元素最小阈值 2.5e3
        max_size: int = self.burnin * 4 # 经验池元素最大阈值 1e4
        self.memory: TensorDictReplayBuffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                max_size=max_size, device=torch.device("cpu")
            )
        ) # 经验池 on CPU

        pre_rate: float = 1 - 1 / (self.max_times * (1 - test_rate) * exploration_rate_min)
        exploration_rate: float = 1 # 基础探索概率
        self.exploration_rate_min: float = exploration_rate_min # 探索概率最小阈值
        self.exploration_rate_decay: float = np.power(self.exploration_rate_min / exploration_rate, 1 / ((1 - pre_rate) * (self.max_times * (1 - test_rate)) * max_size)) # 探索概率衰减率
        exploration_rate_bias: float = exploration_rate / np.power(self.exploration_rate_min / exploration_rate, pre_rate / (1 - pre_rate)) - exploration_rate # 探索率偏置
        self.exploration_rate: float = exploration_rate + exploration_rate_bias
        self.curr_step: int = 0 # 时间步计数

        self.save_every: int = np.abs(max_size * self.max_times).astype(int) // 100 # 模型参数保存频率

        self.sync_every: int = np.rint(max_size * self.max_times * HYPERPARAMETERS["sync_every_rate"]).astype(int) # 参数同步周期

        self.optimizer: torch.optim.Adam = torch.optim.Adam(self.net.parameters(), lr=HYPERPARAMETERS["lr"]) # 使用Adam优化算法

        self.loss_fn: nn.SmoothL1Loss = nn.SmoothL1Loss() # 使用Smooth L1 Loss损失函数

        self.gamma = HYPERPARAMETERS["gamma"] # 未来奖励贴现

    def save(self) -> None:
        """
        将模型参数进行保存（包括探索概率）
        """
        save_path: Path = (
            self.save_dir / f"policy_net.svmd"
        )
        torch.save(
            dict(
                model=self.net.state_dict(),
                variant=self.variant,
                sync_every=self.sync_every,
                optimizer=self.optimizer.state_dict(),
                gamma=self.gamma
            ),
            save_path
        )
        # print(f"PolicyNet saved to {save_path} at step {self.curr_step}")
    
    def loads(self):
        save_path: Path = (
            self.save_dir / f"policy_net.svmd"
        )
        if not save_path.exists(): return False

        save_model = torch.load(save_path, weights_only=False)
        self.net.load_state_dict(save_model['model'])
        self.variant = save_model['variant']
        self.sync_every = save_model['sync_every']
        self.optimizer.load_state_dict(save_model['optimizer'])
        self.gamma = save_model['gamma']
        # print(f"loaded from {save_path}")
        return True

    def update_Q_net(self, td_estimate: torch.Tensor, td_target: torch.Tensor) -> float:
        """
        优化策略
        根据TD估计与TD目标进行反向传播
        TD估计时是给定状态的预测最优Q*
        TD目标是当前奖励和下一个状态s'中估计Q*的聚合
        """
        loss: torch.Tensor = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()
        return loss.cpu().item()
    
    def sync_Q_assignor(self) -> None:
        """
        将online网络参数同步给target网络
        """
        self.net.target.load_state_dict(self.net.online.state_dict())
        # print(f"sync net paramaters at step: {self.curr_step}")

    def act(self, drone_state: np.ndarray, task_state: np.ndarray, action_state: np.ndarray, is_test: bool=False) -> np.ndarray:
        """
        根据无人机状态和任务状态进行决策
        """
        action: torch.Tensor # 分配顺位任务的无人机索引
        if not is_test and np.random.rand() < self.exploration_rate:
            action = self.__explorate_act(drone_state=drone_state, task_state=task_state, action_state=action_state)
        
        else:
            drone_state = torch.tensor(drone_state, device=self.device, dtype=torch.float32)
            task_state = torch.tensor(task_state, device=self.device, dtype=torch.float32)
            action_state = torch.tensor(action_state, device=self.device, dtype=torch.int64)
            _HS, action = self.net(drone_state=drone_state, task_state=task_state, action_state=action_state, model="online")
            action = action.cpu().numpy().astype(np.int64)
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)

        self.curr_step += 1
        return action

    def __explorate_act(self, drone_state: np.ndarray, task_state: np.ndarray, action_state: np.ndarray) -> np.ndarray:
        """
        `探索`
        根据无人机状态与任务状态，将任务做随机分配
        """
        return np.hstack((action_state, np.random.randint(low=0, high=len(drone_state), size=(len(task_state) - len(action_state), ))))

    def cache(self, drone_state: np.ndarray, task_state: np.ndarray, action_state: np.ndarray, next_drone_state: np.ndarray, next_task_state: np.ndarray, action: np.ndarray, reward: float, done: bool) -> None:
        """
        将当前无人机状态、任务状态，下一时间步无人机状态、任务状态，当前行动及其获得的奖励与进展状况放入经验池
        """
        pad_drone = MAX_DRONE_COUNT - len(drone_state)
        pad_task = MAX_TASK_COUNT - len(task_state)
        pad_action = MAX_TASK_COUNT - len(action_state)

        pad_next_drone = MAX_DRONE_COUNT - len(next_drone_state)
        pad_next_task = MAX_TASK_COUNT - len(next_task_state)
        pad_next_action = MAX_TASK_COUNT - len(action)

        assert done and np.array_equal(task_state, next_task_state) or np.array_equal(task_state, next_task_state[:len(action)])
        assert np.array_equal(action_state, action[:len(action_state)])
        
        drone_state = F.pad(torch.tensor(drone_state, dtype=torch.float32), (0, 0, 0, pad_drone), 'constant', torch.nan)
        task_state = F.pad(torch.tensor(task_state, dtype=torch.float32), (0, 0, 0, pad_task), 'constant', torch.nan)
        action_state = F.pad(torch.tensor(action_state, dtype=torch.int64), (0, pad_action), 'constant', -1)
        next_drone_state = F.pad(torch.tensor(next_drone_state, dtype=torch.float32), (0, 0, 0, pad_next_drone), 'constant', torch.nan)
        next_task_state = F.pad(torch.tensor(next_task_state, dtype=torch.float32), (0, 0, 0, pad_next_task), 'constant', torch.nan)
        action = F.pad(torch.tensor(action, dtype=torch.int64), (0, pad_next_action), 'constant', -1)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)
        
        self.memory.add(
            TensorDict(
                {
                    "drone_state": drone_state,
                    "task_state": task_state,
                    "action_state": action_state,
                    "next_drone_state": next_drone_state,
                    "next_task_state": next_task_state,
                    "action": action,
                    "reward": reward,
                    "done": done
                }, batch_size=[]
            )
        )

    def recall(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        `利用`
        从经验池获取数据
        """
        drone_state: torch.Tensor # 当前时间步无人机状态 [n, p1]
        task_state: torch.Tensor # 当前时间步任务状态 [m, p2]
        action_state: torch.Tensor # 方案 [_m]
        next_drone_state: torch.Tensor # 下一时间步无人机状态 [n, p1]
        next_task_state: torch.Tensor # 下一时间步任务状态 [m, p2]
        action: torch.Tensor # 当前时间步采取行动（分配方案）[m,]
        reward: torch.Tensor # 当前时间步获取奖励 []
        done: torch.Tensor # 当前时间步是否结束 bool:[]

        batch = self.memory.sample(1).to(self.device)
        
        drone_state, task_state, action_state, next_drone_state, next_task_state, action, reward, done = (batch.get(key) for key in ("drone_state", "task_state", "action_state", "next_drone_state", "next_task_state", "action", "reward", "done"))
        
        drone_state = drone_state.squeeze(0)
        drone_state = drone_state[~drone_state.isnan().all(dim=1)]
        
        task_state = task_state.squeeze(0)
        task_state = task_state[~task_state.isnan().all(dim=1)]
        
        action_state = action_state.squeeze(0)
        action_state = action_state[(action_state + 1).bool()]

        next_drone_state = next_drone_state.squeeze(0)
        next_drone_state = next_drone_state[~next_drone_state.isnan().all(dim=1)]
        
        next_task_state = next_task_state.squeeze(0)
        next_task_state = next_task_state[~next_task_state.isnan().all(dim=1)]
        
        action = action.squeeze(0)
        action = action[(action + 1).bool()]
        
        reward = reward.squeeze(0)
        
        done = done.squeeze(0)
        
        return drone_state, task_state, action_state, next_drone_state, next_task_state, action, reward, done 

    def td_estimate(self, drone_state: torch.Tensor, task_state: torch.Tensor, action_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        `TD估计`
        根据无人机状态和任务状态估算采取某分配方案的价值
        """
        assert task_state.size(dim=0) == action.size(dim=0)
        assert torch.equal(action[:action_state.size(dim=0)], action_state)
        current_Q, _HS = self.net(drone_state=drone_state, task_state=task_state, action_state=action_state, model="online", action=action)
        assert torch.equal(_HS, action)

        return current_Q

    @torch.no_grad()
    def td_target(self, reward: torch.Tensor, next_drone_state: torch.Tensor, next_task_state: torch.Tensor, next_action_state: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """
        根据下一时间步的无人机状态和任务状态作分配决策，计算其价值估计
        """
        action: torch.Tensor # 做出的决策
        next_Q: torch.Tensor # 决策的价值估计

        _HS, action = self.net(drone_state=next_drone_state, task_state=next_task_state, action_state=next_action_state, model="online")
        assert torch.equal(action[:next_action_state.size(dim=0)], next_action_state)
        next_Q, _HS = self.net(drone_state=next_drone_state, task_state=next_task_state, action_state=next_action_state, model="target", action=action)
        assert torch.equal(_HS, action)
        _d = 1 - done.float()

        return reward + _d * self.gamma * next_Q

    def learn(self) -> tuple[None, None] | tuple[float, float]:
        
        drone_state: torch.Tensor # 当前时间步无人机状态 [n, p1]
        task_state: torch.Tensor # 当前时间步任务状态 [m, p2]
        next_drone_state: torch.Tensor # 下一时间步无人机状态 [n, p1]
        next_task_state: torch.Tensor # 下一时间步任务状态 [m, p2]
        action: torch.Tensor # 当前时间步采取行动（分配方案）[m,]
        reward: torch.Tensor # 当前时间步获取奖励 []
        done: torch.Tensor # 当前时间步是否结束 bool:[]
        td_estimate: torch.Tensor # 动作的价值估计
        td_target: torch.Tensor # 带有未来奖励的价值估计

        if self.curr_step % self.sync_every == 0: # 参数同步
            self.sync_Q_assignor()

        if self.curr_step % self.save_every == 0: # 模型保存
            self.save()

        if self.curr_step < self.burnin: # 积累经验
            return None, None
    
        drone_state, task_state, action_state, next_drone_state, next_task_state, action, reward, done = self.recall()

        td_estimate = self.td_estimate(drone_state=drone_state, task_state=task_state, action_state=action_state, action=action)

        td_target = self.td_target(reward=reward, next_drone_state=next_drone_state, next_task_state=next_task_state, next_action_state=action, done=done)

        loss = self.update_Q_net(td_estimate=td_estimate, td_target=td_target)

        return td_estimate.cpu().item(), loss
        
