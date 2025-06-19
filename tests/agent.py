# %%
import torch
import numpy as np
from pathlib import Path

# %%
from . import SEED

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# %%
from .model.net import PolicyNet

# %%
class Agent:
    def __init__(self, variant: str) -> None:

        self.variant = variant # 变体

        self.device: torch.device = torch.device(torch.cuda.is_available() and "cuda" or "cpu") # 使用gpu或cpu

        self.net: PolicyNet = PolicyNet(variant=variant).float() # 策略模型
        self.net: PolicyNet = self.net.to(device=self.device) # 迁移模型至运算设备

        self.gamma = None # 未来奖励贴现
    
    @classmethod
    def loads(cls, from_path: Path) -> "Agent":
        save_path = from_path / "policy_net.svmd"
        if not save_path.exists(): return False
        
        save_model = torch.load(save_path, weights_only=False)
        
        return cls.load(save_model=save_model)

    @classmethod
    def load(cls, save_model):
        if "variant" in save_model:
            variant = save_model["variant"]
        elif "is_base" in save_model:
            variant = "000" if save_model["is_base"] else "111"
        else:
            raise ValueError(f"illegal save_model, there is no is_base or variant")
        agent = cls(variant=variant)
        agent.net.load_state_dict(save_model["model"])
        agent.gamma = save_model["gamma"]
        return agent

    @torch.no_grad()
    def act(self, drone_state: np.ndarray, task_state: np.ndarray, action_state: np.ndarray) -> tuple[float, np.ndarray]:
        """
        根据无人机状态和任务状态进行决策
        """
        drone_state = torch.tensor(drone_state, device=self.device, dtype=torch.float32)
        task_state = torch.tensor(task_state, device=self.device, dtype=torch.float32)
        action_state = torch.tensor(action_state, device=self.device, dtype=torch.int64)
        q, action = self.net(drone_state=drone_state, task_state=task_state, action_state=action_state, model="online")
        action = action.cpu().numpy().astype(np.int64)

        return q.cpu().item(), action
