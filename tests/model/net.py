# %%
from . import SEED
from ..env import DRONE_FEATURES_SIZE, TASK_FEATURES_SIZE

# %%
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

# %%
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# %%
class PolicyNet(nn.Module):
    def __init__(self, variant: str) -> None:
        super().__init__()
        
        self.online = _build_net(variant=variant)
        self.target = _build_net(variant=variant)
        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, model: str, **kwargs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if model == "online":
            return self.online(**kwargs)
        elif model == "target":
            return self.target(**kwargs)

class _build_net(nn.Module):
    def __init__(self, variant: str) -> None:
        super().__init__()
        
        is_base_encoder, is_base_Q, is_base_updater = list(map(lambda flag: 1 ^ int(flag), variant))

        if not is_base_encoder:
            self.shared_module = nn.Linear(in_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE, out_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE)
        self.drone_encoder: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=DRONE_FEATURES_SIZE, out_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE),
            nn.ReLU(),
            self.shared_module if not is_base_encoder else nn.Linear(in_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE, out_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE, out_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE),
            nn.ReLU()
        )
        self.task_encoder: nn.Sequential =  nn.Sequential(
            nn.Linear(in_features=TASK_FEATURES_SIZE, out_features=TASK_FEATURES_SIZE * DRONE_FEATURES_SIZE),
            nn.ReLU(),
            self.shared_module if not is_base_encoder else nn.Linear(in_features=TASK_FEATURES_SIZE * DRONE_FEATURES_SIZE, out_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE, out_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE),
            nn.ReLU(),
        )
        self.Q_net: _q_net = _q_net(is_base=is_base_Q)
        self.updater: _gate_updater = _gate_updater(is_base=is_base_updater)
        
    def forward(self, drone_state: torch.Tensor, task_state: torch.Tensor, action_state: torch.Tensor, action: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        `drone_state`: 无人机状态 [n, p1]
        `task_state`: 任务状态 [m, p2]
        `action_state`: 已分配方案 [m', ]
            m'：为已分配方案中的任务数目
            该矩阵元素为无人机索引：A[i]=j 表示将任务i分配给无人机jdx_drone
        返回 价值估计与决策方案
        """
        drone_size: int # 参与分配的无人机数目n
        task_size: int # 参与分配的任务数目m
        action_state: torch.Tensor
        mask: torch.Tensor # 掩码，分配方案
        drone_embeddings: torch.Tensor # 无人机编码 [n, p1 * p2]
        task_embeddings: torch.Tensor # 任务编码 [m, p2 * p1]
        Q: list | torch.Tensor # 部分价值估计 [m - m',]
        q: torch.Tensor # 局部价值估计 [n, m, 1] -> [n, m]

        drone_size, task_size = drone_state.size(dim=0), task_state.size(dim=0)

        drone_embeddings = self.drone_encoder(drone_state)
        task_embeddings = self.task_encoder(task_state)

        flag = action_state.size(dim=0)

        mask = -torch.ones(size=(task_size, task_size), device=action_state.device)
        Q = 0
        for idx in range(task_size):
            q = self.Q_net(drone_embeddings, task_embeddings)
            q = q.squeeze(dim=-1)

            done_tasks = (mask[:idx] + 1).bool().any(dim=0)
            q = q.masked_fill(mask=done_tasks.expand_as(q), value=-torch.inf)
            
            qf = q.flatten()
            p = qf.softmax(dim=-1)
            
            selected = p.argmax()
            selected_task = (selected % task_size).int()
            selected_drone = (action_state[selected_task] if selected_task < flag else selected // task_size if action == None else action[selected_task]).int()
            Q = Q + q[selected_drone, selected_task]

            if selected_task < flag:
                Q = Q - q[selected_drone, selected_task]
            
            mask[idx, selected_task] = selected_drone

            new_drone_embedding = self.updater(drone_embeddings[selected_drone], task_embeddings[selected_task])

            drone_embeddings = drone_embeddings.index_put(
                indices=(selected_drone, ),
                values=new_drone_embedding
            )
        
        action, _ = mask.max(dim=0)
        return Q, action

class _q_net(nn.Module):
    def __init__(self, is_base: bool):
        super().__init__()
        self.td2s2 = _multi_scale(out_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE * 2, is_base=is_base)
        self.td2s4 = _multi_scale(out_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE * 4, is_base=is_base)
        self.s62s6 = _multi_scale(out_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE * 6, is_base=is_base)

        self.s62h = nn.Sequential(
            nn.Linear(in_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE * 6, out_features=(DRONE_FEATURES_SIZE + TASK_FEATURES_SIZE) * 6),
            nn.ReLU(),
            nn.Linear(in_features=(DRONE_FEATURES_SIZE + TASK_FEATURES_SIZE) * 6, out_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE * 6),
            nn.ReLU()
        )
        
        self.h2a = nn.Linear(in_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE * 6, out_features=1)
        self.h2v = nn.Linear(in_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE * 6, out_features=1)
        
    def forward(self, drone_embeddings: torch.Tensor, task_embeddings: torch.Tensor) -> torch.Tensor:
        drone_size, task_size = drone_embeddings.size(dim=0), task_embeddings.size(dim=0)
        s2 = self.td2s2(drone_embeddings.unsqueeze(1).expand(-1, task_size, -1), task_embeddings.unsqueeze(0).expand(drone_size, -1, -1))
        s4 = self.td2s4(drone_embeddings.unsqueeze(1).expand(-1, task_size, -1), task_embeddings.unsqueeze(0).expand(drone_size, -1, -1))
        s6 = self.s62s6(s2, s4)
        h = self.s62h(s6)
        a = self.h2a(h)
        v = self.h2v(h)
        return v + a - a.mean()

class _multi_scale(nn.Module):
    def __init__(self, out_features: int, is_base: bool) -> None:
        super().__init__()
        assert out_features % 2 == 0
        self.is_base = is_base
        self.x12s1 = nn.LazyLinear(out_features=out_features // 2)
        self.x22s2 = nn.LazyLinear(out_features=out_features // 2)
        if not is_base:
            self.h2q = nn.Linear(in_features=out_features, out_features=out_features)
            self.h2k = nn.Linear(in_features=out_features, out_features=out_features)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor
        q: torch.Tensor
        k: torch.Tensor
        w: torch.Tensor

        s1 = self.x12s1(x1)
        s2 = self.x22s2(x2)
        h = F.relu(torch.cat((s1, s2), dim=-1))
        
        if self.is_base:
            return h
        
        q = self.h2q(h)
        k = self.h2k(h)
        w = F.softmax(torch.bmm(q, k.transpose(1, 2))/np.sqrt(h.size(-1)), dim=-1)
        return torch.bmm(w, h)

class _gate_updater(nn.Module):
    def __init__(self, is_base: bool):
        super().__init__()
        self.is_base = is_base
        self.x12h1 = nn.LazyLinear(out_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE)
        self.x22h2 = nn.LazyLinear(out_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE)
        self.gate = nn.Sequential(
            nn.Linear(in_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE * 2, out_features=DRONE_FEATURES_SIZE * TASK_FEATURES_SIZE),
            nn.Sigmoid() if not is_base else nn.ReLU()
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.x12h1(x1))
        h2 = F.relu(self.x22h2(x2))
        z = self.gate(torch.cat((h1, h2), dim=-1))
        return z * h1 + (1 - z) * h2 if not self.is_base else z