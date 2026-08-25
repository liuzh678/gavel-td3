# -*- coding: utf-8 -*

from copy import deepcopy
import os
import time
import math
import numpy as np
import random
import gc, torch
import torch.nn as nn
import queue
import sys
import torch.nn.functional as F
from torch_geometric.data import Batch
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from collections import deque
import torch.multiprocessing as mp #A
from replay_buffer_graph import ReplayBuffer
from velodyne_env import GazeboEnv
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_undirected, remove_self_loops

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_ROOT = os.path.abspath(
    os.environ.get("GAVEL_OUTPUT_DIR", os.path.join(REPO_ROOT, "outputs"))
)
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
FINAL_MODEL_DIR = os.path.join(OUTPUT_ROOT, "models")
TENSORBOARD_DIR = os.path.join(OUTPUT_ROOT, "tensorboard")
# —— 完全图 edge_index：21×21，只生成一次 —— 
# =============================================================

# ────────────────── 全局常量 ──────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(sci_mode=False, precision=2)
INITIAL_EVAL_EPISODES = 50
N_NODES = 21                                         # 20 LiDAR + 1 Robot
robot_idx = torch.full((20,), 20)
lidar_idx = torch.arange(20)

# ===== 新增：LiDAR–LiDAR 完全子图（去自环，先加一半，等会儿 to_undirected 变成双向）=====
li_i, li_j = torch.meshgrid(lidar_idx, lidar_idx, indexing='ij')
mask = (li_i < li_j)                                  # 只取上三角，避免自环与重复
ll_row = li_i[mask].reshape(-1)                       # 190 条（无向）
ll_col = li_j[mask].reshape(-1)

# 你原有的 LiDAR <-> Robot 双向
row = torch.cat([robot_idx, lidar_idx, ll_row])
col = torch.cat([lidar_idx, robot_idx, ll_col])

edge_index = torch.stack([row, col], dim=0)
edge_index = to_undirected(edge_index, num_nodes=N_NODES)  # 现在 LL 变成双向，共 380 条

# 仍然可以移除自环（此时只有可能误删 robot 自环）
edge_index, _ = remove_self_loops(edge_index)

# ===== 新增：补回 Robot–Robot 自环 =====
loop = torch.tensor([[20], [20]], device=edge_index.device, dtype=edge_index.dtype)
edge_index = torch.cat([edge_index, loop], dim=1)

EDGE_INDEX_CONST = edge_index                         # 期望形状：(2, 421)



# 20 条分界：-90° … +90°
N = 20  # = 20
left0 = -torch.pi/2 - 0.03
w = torch.pi / N

left_edges  = left0 + w * torch.arange(N, device=device)     # [N]
right_edges = left_edges + w                                  # [N]
right_edges[-1] = right_edges[-1] + 0.03                     # 末段 +0.03，和你 gaps 保持一致

centers = (left_edges + right_edges) / 2                     # [N]
ANGLE_FEAT_CONST = torch.stack([torch.sin(centers), torch.cos(centers)], dim=1)  # [N,2]



class GATEncoder(nn.Module):
    def __init__(self,
                 in_dim=7,
                 embed_dim=128,      # “每节点最终特征维度 d_out”（两层后）
                 heads1=4,           # 第1层多头数
                 heads2=4,           # 第2层多头数（concat=False 时输出仍是 embed_dim）
                 dropout=0.1,
                 add_self_loops=True,
                 output_mode="env_raw_robot"):
        super().__init__()
        self.num_nodes = 21
        self.robot_idx = 20
        self.dropout = dropout
        self.output_mode = output_mode

        # ---------- 第1层 GAT：concat=True（先“变宽”） ----------
        # conv1 输出维：d1 = heads1 * embed_dim
        self.conv1 = GATv2Conv(
            in_dim, embed_dim, heads=heads1,
            concat=True, dropout=dropout, add_self_loops=add_self_loops
        )
        self.d1 = embed_dim * heads1
        self.res1  = nn.Linear(in_dim, self.d1, bias=False)  # 残差对齐维度
        self.norm1 = nn.LayerNorm(self.d1)

        # ---------- 第2层 GAT：concat=False（压回固定宽度 embed_dim） ----------
        # 无论 heads2 多少，conv2 输出维均为 embed_dim
        self.conv2 = GATv2Conv(
            self.d1, embed_dim, heads=heads2,
            concat=False, dropout=dropout, add_self_loops=add_self_loops
        )
        self.res2  = nn.Linear(self.d1, embed_dim, bias=False)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, data):
        """
        输入:
            data.x:           [(B*21), in_dim]
            data.edge_index:  [2, E]
            data.batch(可选): [(B*21)], 值域 0..B-1
        输出:
            S_GAT: [B, 2*embed_dim]
        """
        x0 = data.x                                         # [(B*21), in_dim]
        edge_index = data.edge_index
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = x0.new_zeros(x0.size(0), dtype=torch.long)
        B = int(batch.max().item()) + 1

        # ===== 第1层 =====
        x1 = self.conv1(x0, edge_index)                    # [(B*21), d1]      d1 = heads1 * embed_dim
        x1 = F.elu(x1)
        x1 = self.norm1(x1 + self.res1(x0))                # [(B*21), d1]
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # ===== 第2层 =====
        x2 = self.conv2(x1, edge_index)                    # [(B*21), embed_dim]
        x2 = F.elu(x2)
        x2 = self.norm2(x2 + self.res2(x1))                # [(B*21), embed_dim]
        x  = F.dropout(x2, p=self.dropout, training=self.training)

        # ===== 掩码（固定 21 节点，机器人是 idx=20）=====
        pos = torch.arange(x.size(0), device=x.device, dtype=torch.long) % self.num_nodes
        mask_env   = (pos < self.robot_idx)                # [(B*21),]  选 0..19
        mask_robot = (pos == self.robot_idx)               # [(B*21),]  选 20

        # ===== 池化 =====
        if self.output_mode == "legacy_robot_global":
            xB = x.view(B, self.num_nodes, -1)
            robot_feat = xB[:, self.robot_idx, :]                         # [B, embed_dim]
            global_feat = xB.mean(dim=1)                                  # [B, embed_dim]
            return torch.cat([robot_feat, global_feat], dim=-1)           # [B, 2*embed_dim]

        z_env = global_mean_pool(x[mask_env], batch[mask_env])             # [B, embed_dim]
        # 方式B：按固定顺序直接索引（更高效）
        x0B   = x0.view(B, self.num_nodes, -1)
        h_r0  = x0B[:, self.robot_idx, :]                          # [B, 7]

        S_GAT = torch.cat([z_env, h_r0], dim=-1)
        return S_GAT





        # x = F.elu(self.conv(data.x, data.edge_index))  
        # # 直接全局平均池化到图级特征： (B, embed_dim*heads)
        # return global_mean_pool(x, data.batch)


        # h = F.elu(self.conv(data.x, data.edge_index))
        # B = int(data.batch.max().item()) + 1
        # Fdim = h.size(-1)
        # h = h.view(B, self.num_nodes, Fdim)  # [B, 21, F]
        # robot_feat = h[:, self.robot_idx, :]
        # lidar_mask = torch.ones(self.num_nodes, dtype=torch.bool, device=h.device)
        # lidar_mask[self.robot_idx] = False
        # lidar_feat = h[:, lidar_mask, :]                     # [B, 20, F]

        # lidar_mean = lidar_feat.mean(dim=1)                  # [B, F]
        # lidar_max  = lidar_feat.max(dim=1).values            # [B, F]  # 可选但很有用

        # # 3. 拼接（robot + mean [+ max]）
        # joint_feat = torch.cat([robot_feat, lidar_mean, lidar_max], dim=-1)  # [B, 3F]
        return joint_feat
        # Fdim = h.size(-1)
        # h = h.view(B, self.num_nodes, Fdim)  # [B, 21, F]

        # # 1. 单独提取 robot 节点特征: [B, F]
        # robot_feat = h[:, self.robot_idx, :]

        # # 2. 对所有节点做全局平均池化: [B, F]
        # # 方法1：直接 mean
        # global_feat = h.mean(dim=1)
        # # 方法2（可选）：如果有 batch，可用global_mean_pool
        # # global_feat = global_mean_pool(h.reshape(B*self.num_nodes, Fdim), torch.arange(B).repeat_interleave(self.num_nodes).to(h.device))

        # # 3. 拼接
        # joint_feat = torch.cat([robot_feat, global_feat], dim=-1)  # [B, 2F]

        return joint_feat


class LegacySingleGATEncoder(nn.Module):
    def __init__(self, in_dim=7, embed_dim=128, heads=1, dropout=0.1):
        super().__init__()
        self.num_nodes = 21
        self.robot_idx = 20
        self.conv = GATv2Conv(in_dim, embed_dim, heads=heads, concat=True, dropout=dropout)

    def forward(self, data):
        x = F.elu(self.conv(data.x, data.edge_index))
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        B = int(batch.max().item()) + 1
        xB = x.view(B, self.num_nodes, -1)
        robot_feat = xB[:, self.robot_idx, :]
        global_feat = xB.mean(dim=1)
        return torch.cat([robot_feat, global_feat], dim=-1)


# ── Actor：直接把 Encoder + MLP 定义在一起 ────────────────────────────
# ── Actor：直接把 Encoder + MLP 定义在一起 ────────────────────────────
class Actor(nn.Module):
    def __init__(self,feat_dim: int, action_dim: int, max_action: float, input_dim=None):
        super().__init__()
     #   self.encoder    = encoder
        self.max_action = max_action
        # MLP Head
        self.fc1 = nn.Linear(input_dim if input_dim is not None else feat_dim+7, 800)
        self.fc2 = nn.Linear(800, 600)
        self.fc3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

        # 用于构图：常量 buffer
        self.register_buffer("edge_index", EDGE_INDEX_CONST.long())
        self.register_buffer("angle_feat", ANGLE_FEAT_CONST.float())

    @torch.no_grad()
    def _build_graph(self, state24):
        # 与你原逻辑一致的单条 / 批量图构建
        if isinstance(state24, torch.Tensor) and state24.dim() == 1:
            flat = state24
            laser = flat[:20].unsqueeze(1)
            angle = self.angle_feat.to(device=flat.device)
            zeros3 = torch.zeros(20, 4, device=flat.device)
            sector = torch.cat([laser, angle, zeros3], dim=1)

            robot = flat[20:].unsqueeze(0)
            zeros1 = torch.zeros(1, 3, device=flat.device)
            robot = torch.cat([zeros1, robot], dim=1)

            x = torch.cat([sector, robot], dim=0)
            batch = torch.zeros(21, dtype=torch.long, device=flat.device)
            ei = self.edge_index.to(flat.device)
            return Data(x=x, edge_index=ei, batch=batch)

        # 批量情况
        B = state24.size(0)
        laser  = state24[:, :20].unsqueeze(-1)
        angle  = self.angle_feat.to(state24.device).unsqueeze(0).expand(B, -1, -1)
        zeros3 = torch.zeros(B, 20, 4, device=state24.device)
        sector = torch.cat([laser, angle, zeros3], dim=-1)

        robot  = state24[:, 20:].unsqueeze(1)
        zeros1 = torch.zeros(B, 1, 3, device=state24.device)
        robot  = torch.cat([zeros1, robot], dim=-1)

        x = torch.cat([sector, robot], dim=1).view(B*21, 7)
        E = self.edge_index.size(1)
        ei = self.edge_index.to(state24.device)         # [2,E] long
        ei_rep = ei.repeat(1, B)                         # [2,B*E]
        offsets = (torch.arange(B, device=state24.device) * 21).repeat_interleave(E)  # [B*E]
        eiB = ei_rep + torch.stack([offsets, offsets], dim=0)  # [2, B*E]

        batch = torch.repeat_interleave(torch.arange(B, device=state24.device), 21)
        return Data(x=x, edge_index=eiB, batch=batch)

    def forward(self, feat: torch.Tensor):
        """给定已经编码好的图级特征 feat，计算最终动作输出"""
        h = F.relu(self.fc1(feat))
        h = F.relu(self.fc2(h))
        return self.max_action * self.tanh(self.fc3(h))

    def compute_action_from_feat(self, feat: torch.Tensor):
        """给定已经编码好的图级特征 feat，计算最终动作输出"""
        h = F.relu(self.fc1(feat))
        h = F.relu(self.fc2(h))
        return self.max_action * self.tanh(self.fc3(h))



class Critic(nn.Module):
    def __init__(self,feat_dim: int, action_dim: int, input_dim=None):
        super().__init__()
   #     self.encoder = encoder

        # 双 Q 网络
        self.q1_fc1 = nn.Linear(input_dim if input_dim is not None else feat_dim+7, 800)
        self.q1_fc2 = nn.Linear(800, 600)
        self.q1_fc_a = nn.Linear(action_dim, 600)
        self.q1_out = nn.Linear(600, 1)

        self.q2_fc1 = nn.Linear(input_dim if input_dim is not None else feat_dim+7, 800)
        self.q2_fc2 = nn.Linear(800, 600)
        self.q2_fc_a = nn.Linear(action_dim, 600)
        self.q2_out = nn.Linear(600, 1)

        # 构图所需常量
        self.register_buffer("edge_index", EDGE_INDEX_CONST.long())
        self.register_buffer("angle_feat", ANGLE_FEAT_CONST.float())

    @torch.no_grad()
    def _build_graph(self, state24):
        # 与 Actor 完全相同
        if isinstance(state24, torch.Tensor) and state24.dim() == 1:
            flat = state24
            laser = flat[:20].unsqueeze(1)
            angle = self.angle_feat
            zeros3 = torch.zeros(20, 4, device=flat.device)
            sector = torch.cat([laser, angle, zeros3], dim=1)

            robot = flat[20:].unsqueeze(0)
            zeros1 = torch.zeros(1, 3, device=flat.device)
            robot = torch.cat([zeros1, robot], dim=1)

            x = torch.cat([sector, robot], dim=0)
            batch = torch.zeros(21, dtype=torch.long, device=flat.device)
            ei = self.edge_index.to(flat.device)
            return Data(x=x, edge_index=ei, batch=batch)

        B = state24.size(0)
        laser  = state24[:, :20].unsqueeze(-1)
        angle  = self.angle_feat.to(state24.device).unsqueeze(0).expand(B, -1, -1)
        zeros3 = torch.zeros(B, 20, 4, device=state24.device)
        sector = torch.cat([laser, angle, zeros3], dim=-1)

        robot  = state24[:, 20:].unsqueeze(1)
        zeros1 = torch.zeros(B, 1, 3, device=state24.device)
        robot  = torch.cat([zeros1, robot], dim=-1)

        x = torch.cat([sector, robot], dim=1).view(B*21, 7)
        E = self.edge_index.size(1)
        ei = self.edge_index.to(state24.device)          # [2,E] long
        ei_rep = ei.repeat(1, B)                         # [2,B*E]
        offsets = (torch.arange(B, device=state24.device) * 21).repeat_interleave(E)  # [B*E]
        eiB = ei_rep + torch.stack([offsets, offsets], dim=0)  # [2, B*E]
        batch = torch.repeat_interleave(torch.arange(B, device=state24.device), 21)
        return Data(x=x, edge_index=eiB, batch=batch)

    def forward(self, feat, action):
        """已编码特征 + 动作，计算 Q1 Q2"""
        q1 = self._forward_q1(feat, action)
        q2 = self._forward_q2(feat, action)
        return q1, q2

    def _forward_q1(self, feat, action):
        h_s = F.relu(self.q1_fc1(feat))
        h_s = self.q1_fc2(h_s)
        h_a = self.q1_fc_a(action)
        return self.q1_out(F.relu(h_s + h_a))

    def _forward_q2(self, feat, action):
        h_s = F.relu(self.q2_fc1(feat))
        h_s = self.q2_fc2(h_s)
        h_a = self.q2_fc_a(action)
        return self.q2_out(F.relu(h_s + h_a))

    def compute_Q_from_feat(self, feat, action):
        """已编码特征 + 动作，计算 Q1 Q2"""
        q1 = self._forward_q1(feat, action)
        q2 = self._forward_q2(feat, action)
        return q1, q2




class OUNoise:
    def __init__(self, action_dim=2, mu=0.0, theta=0.00006, sigma=1, sigma_min=0.15, sigma_decay_steps=30000):
        """
        初始化 OU 噪声参数
        :param action_dim: 动作的维度
        :param mu: 噪声的均值，默认值为0
        :param theta: 噪声向均值收敛的速度参数，默认值为0.2（较大以增强回归趋势）
        :param sigma: 初始噪声的波动幅度，默认值为0.2（较小以减少随机波动）
        :param sigma_min: 最小噪声波动幅度，默认值为0.05
        :param sigma_decay_steps: 噪声波动幅度的衰减步数
        """
        self.action_dim = action_dim
        self.mu = mu  # 噪声的均值
        self.theta = theta  # 噪声向均值收敛的速度
        self.sigma = sigma  # 初始噪声波动幅度
        self.sigma_min = sigma_min  # 最小噪声波动幅度
        self.sigma_decay = (sigma - sigma_min) / sigma_decay_steps  # 每步的噪声衰减量
        self.state = np.ones(self.action_dim) * self.mu  # 初始化状态

    def reset(self):
        """
        重置噪声状态为均值
        """
        self.state = np.ones(self.action_dim) * self.mu
        self.decay_sigma()  # 在每个 Episode 结束时衰减 sigma

    def evolve_state(self):
        """
        更新噪声状态
        :return: 更新后的噪声状态
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=self.action_dim)
        x = x + dx
        x = np.tanh(x)  # 将值压缩到 (-1, 1)
        self.state = (x + 1) / 2  # 映射到 [0, 1]
        return self.state

    def get_noise(self):
        """
        获取当前噪声值，并衰减噪声幅度
        :return: 当前噪声值
        """
        noise = self.evolve_state()

        return noise

    def decay_sigma(self):
        """
        衰减噪声的波动幅度，直到达到最小值
        """
        if self.sigma > self.sigma_min:
            self.sigma -= self.sigma_decay
            self.sigma = max(self.sigma, self.sigma_min)
            
            

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action,
                 embed_dim=128, heads=2, legacy_feature_256=False, legacy_single_gat=False):
        # 1) 构建共享 Encoder 及其 target 版
        if legacy_single_gat:
            self.encoder = LegacySingleGATEncoder(in_dim=7, embed_dim=embed_dim, heads=heads).to(device)
        else:
            encoder_output_mode = "legacy_robot_global" if legacy_feature_256 else "env_raw_robot"
            self.encoder = GATEncoder(in_dim=7, embed_dim=embed_dim, heads1=heads,heads2=heads, output_mode=encoder_output_mode).to(device)
      #  self.encoder_target = deepcopy(self.encoder).to(device)
        
        feat_dim = embed_dim 
        policy_input_dim = embed_dim * 2 if legacy_feature_256 else None
        
        # 2) Actor & Actor_target
        self.actor = Actor( feat_dim, action_dim, max_action, input_dim=policy_input_dim).to(device)
        self.actor_target = Actor( feat_dim, action_dim, max_action, input_dim=policy_input_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # 3) Critic & Critic_target
        self.critic = Critic(feat_dim, action_dim, input_dim=policy_input_dim).to(device)
        self.critic_target = Critic( feat_dim, action_dim, input_dim=policy_input_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.max_action = max_action
        self.iter_count = 0

    def share_memory(self):
        self.encoder.share_memory()
      #  self.encoder_target.share_memory()
        self.actor.share_memory()
        self.actor_target.share_memory()
        self.critic.share_memory()
        self.critic_target.share_memory()

    @torch.no_grad()
    def get_action(self, state):
        # state: (state_dim,) numpy 或 Tensor
        state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=device)

        # 1. 构建图（你自己定义的函数或模块）
        data = self.actor._build_graph(state_tensor)         # 输出应为 torch_geometric.data.Data
     

        # 2. 用 encoder 提取图特征
        feat = self.encoder(data)                     # 输出维度: (1, feat_dim)

        # 3. 用 actor 得到动作
        action = self.actor.compute_action_from_feat(feat)                     # 输出维度: (1, action_dim)

        return action.cpu().numpy().flatten()

    # training cycle
 
    def train(
            self,
            replay_buffer,
            iterations,
            #writer,
            Critic_Loss_queue,
            Actor_Loss_queue,
            dccs_avg_queue,
            dccs_std_queue,
            Av_Q_queue,
            Max_Q_queue,
            counter_epoch,
            batch_size=40,
            discount=1,
            tau=0.005,
            policy_noise=0.2,  # discount=0.99
            noise_clip=0.5,
            policy_freq=2,
            Lamda=0,
            a_l=0,
        ):
            critic_lr = 1e-3 # Critic的学习率
            actor_lr = 4e-4 # Actor的学习率
            encoder_lr=3e-4
            # critic_lr = 7e-4   # 或按上面的档位替换
            # actor_lr  = 2e-4
            # encoder_lr = 1e-4
            print("actor_lr:",actor_lr)
            critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
            actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
            encoder_optimizer = torch.optim.Adam(self.encoder.parameters(),lr=encoder_lr)
            av_Q = 0
            max_Q = -inf
            c_av_loss = 0
            a_av_loss = 0
            for it in range(iterations):
                # sample a batch from the replay buffer
                (
                    batch_states,
                    batch_actions,
                    batch_rewards,
                    batch_dones,
                    batch_next_states,
                    batch_U_s,
                    batch_omega,
                ) = replay_buffer.sample_batch(batch_size)
                state = torch.Tensor(batch_states).to(device)
                next_state = torch.Tensor(batch_next_states).to(device)
                action = torch.Tensor(batch_actions).to(device)
                reward = torch.Tensor(batch_rewards).to(device)
                done = torch.Tensor(batch_dones).to(device)
                U_s=torch.Tensor(batch_U_s).to(device)
                omega=torch.Tensor(batch_omega).to(device)
        
                with torch.no_grad():
                    data_next  = self.actor_target._build_graph(next_state)
                    feat_next  = self.encoder(data_next)
                    next_action = self.actor_target.compute_action_from_feat(feat_next)

                    noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
                    noise = noise.clamp(-noise_clip, noise_clip)
                    next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                target_Q1, target_Q2 = self.critic_target.compute_Q_from_feat(feat_next, next_action)

                target_Q = torch.min(target_Q1, target_Q2)
    
                av_Q += torch.mean(target_Q)
                max_Q = max(max_Q, torch.max(target_Q))
                # # Calculate the final Q value from the target network parameters by using Bellman equation
                target_Q = reward + ((1 - done) * discount * target_Q).detach()

                # Get the Q values of the basis networks with the current parameters
                next_data_next  = self.actor_target._build_graph(state)
                next_feat_next  = self.encoder(next_data_next)
                current_Q1, current_Q2= self.critic.compute_Q_from_feat(next_feat_next, action)



        
                loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

                
                critic_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                loss.backward()
                critic_optimizer.step()
                encoder_optimizer.step()

                if it % policy_freq == 0:
                    # 1. 只构图 & 编码一次
                    data = self.actor._build_graph(state)
                    feat = self.encoder(data)

                    # 2. 计算当前 actor 输出
                    predicted_action = self.actor.compute_action_from_feat(feat)

                    actor_grad, _ = self.critic.compute_Q_from_feat(feat, predicted_action)
                    q_PF = ((1 - torch.cos(predicted_action[:, 0] - U_s)) + (1 - torch.cos(predicted_action[:, 1] - omega)))
        
                    actor_grad_combined = (-(1 - Lamda) * actor_grad + Lamda * q_PF).mean()  # 结合APF和TD3，并转换为标量

                    actor_optimizer.zero_grad()
                #  encoder_optimizer.zero_grad()
                    actor_grad_combined.backward()
                    actor_optimizer.step()
                #  encoder_optimizer.step() 
                    # Use soft update to update the actor-target network parameters by
                    # infusing small amount of current parameters
                    for param, target_param in zip(
                        self.actor.parameters(), self.actor_target.parameters()
                    ):
                        target_param.data.copy_(
                            tau * param.data + (1 - tau) * target_param.data
                        )
                    # Use soft update to update the critic-target network parameters by infusing
                    # small amount of current parameters
                    for param, target_param in zip(
                        self.critic.parameters(), self.critic_target.parameters()
                    ):
                        target_param.data.copy_(
                            tau * param.data + (1 - tau) * target_param.data
                        )
                        
                    # for param, target_param in zip(self.encoder.parameters(), self.encoder_target.parameters()):
                    #     target_param.data.copy_(
                    #         tau * param.data + (1 - tau) * target_param.data
                    #     )
                c_av_loss += loss
                a_av_loss+=actor_grad_combined
            self.iter_count += 1
            # Write new values for tensorboard
            Critic_Loss_queue.put(( (1.0*c_av_loss / iterations).detach(), counter_epoch))
            Actor_Loss_queue.put(( (1.0*a_av_loss / iterations).detach(), counter_epoch))
            Av_Q_queue.put(( (1.0*av_Q / iterations).detach(), counter_epoch))
            Max_Q_queue.put(( (1.0*max_Q).detach(), counter_epoch))   
            # writer.add_scalar("Critic-Loss", c_av_loss / iterations, counter_epoch)
            # writer.add_scalar("Actor-Loss", a_av_loss / iterations,counter_epoch)
            # writer.add_scalar("Av. Q", av_Q / iterations, counter_epoch)
            # writer.add_scalar("Max. Q", max_Q, counter_epoch)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        torch.save(self.encoder.state_dict(),
                   f"{directory}/{filename}_encoder.pth")

    def load(self, filename, directory):
        self.encoder.load_state_dict(
            torch.load(f"{directory}/{filename}_encoder.pth", map_location=device)
        )
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename), map_location=device)
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename), map_location=device)
        )
      #  self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.actor_target  .load_state_dict(self.actor.state_dict())
        self.critic_target .load_state_dict(self.critic.state_dict())


def calculate_apf_reference(env, log_label):
    odom = env.last_odom
    if odom is None:
        odom = env.wait_for_odom(timeout=30.0, log_label=log_label)
    robot_x = odom.pose.pose.position.x
    robot_y = odom.pose.pose.position.y
    p = env.calculate_attraction_force(robot_x, robot_y, env.goal_x, env.goal_y, 650)
    f = env.calculate_total_repulsion_force(robot_x, robot_y, math.sqrt(p[0] ** 2 + p[1] ** 2))
    z = (f[0] + p[0], f[1] + p[1])
    v = math.sqrt(z[0] ** 2 + z[1] ** 2)
    v = min(v, 0.5)
    v = max(v, 0)
    omega = math.atan2(z[1], z[0])
    omega = env.angle - omega

    if omega > 3.1:
        omega = 6.2 - omega
        omega = -omega
    if omega < -3.1:
        omega = omega + 6.2
    omega1 = omega
    omega = min(omega, 1)
    omega = max(omega, -1)
    omega = -omega
    return v, omega, omega1


def linear_decay_to_zero(initial_value, elapsed_episodes, decay_episodes):
    if decay_episodes <= 0:
        return 0.0
    progress = min(max(float(elapsed_episodes), 0.0), float(decay_episodes))
    return float(initial_value) * (1.0 - progress / float(decay_episodes))


def run_initial_eval_zero_window(
    network,
    env,
    max_ep,
    max_action,
    success_queue,
    reward_queue,
    reward_all,
    timestep_episode,
):
    np_random_state = np.random.get_state()
    py_random_state = random.getstate()
    env_state = {
        "upper": env.upper,
        "lower": env.lower,
        "last_distance": env.last_distance,
        "last_speed": env.last_speed,
        "last_angular_velocity": env.last_angular_velocity,
        "succes": env.succes,
        "end_check": env.end_check,
        "collision_check": env.collision_check,
        "choice": env.choice,
    }
    encoder_was_training = network.encoder.training
    actor_was_training = network.actor.training

    success_count = 0
    network.encoder.eval()
    network.actor.eval()
    print(f"Running initial zero-window evaluation for {INITIAL_EVAL_EPISODES} episodes")

    try:
        for eval_idx in range(INITIAL_EVAL_EPISODES):
            print(f"initial_eval_episode={eval_idx + 1}/{INITIAL_EVAL_EPISODES}")
            state = env.reset()
            env.choice = True
            done = False
            episode_reward = 0.0
            episode_timesteps = 0
            episode_success = False

            while not done and episode_timesteps < max_ep:
                _, _, omega1 = calculate_apf_reference(env, "/odom before initial eval APF")
                action = network.get_action(np.array(state))
                action = np.clip(action, -max_action, max_action)
                a_in = [(action[0] + 1) / 2, action[1]]
                next_state, reward, done, target = env.step(state, a_in, omega1)
                episode_reward += reward
                episode_timesteps += 1
                state = next_state
                if target:
                    episode_success = True
                if episode_timesteps >= max_ep:
                    done = True

            if episode_timesteps <= 0:
                raise RuntimeError("initial evaluation episode_timesteps <= 0")

            if episode_success or env.end_check:
                success_count += 1
            reward_queue.put((episode_reward / episode_timesteps, eval_idx))
            reward_all.put((episode_reward, eval_idx))
            timestep_episode.put((episode_timesteps, eval_idx))
            env.end_check = False
            env.collision_check = False

        success_queue.put((success_count / INITIAL_EVAL_EPISODES, 0))
        print(
            "Initial zero-window evaluation done: "
            f"success_rate={success_count / INITIAL_EVAL_EPISODES}"
        )
    finally:
        if encoder_was_training:
            network.encoder.train()
        else:
            network.encoder.eval()
        if actor_was_training:
            network.actor.train()
        else:
            network.actor.eval()
        for name, value in env_state.items():
            setattr(env, name, value)
        np.random.set_state(np_random_state)
        random.setstate(py_random_state)


def worker(t,network,counter_epoch,counter_epoch_success,param,success_queue,reward_queue,Critic_Loss_queue,Actor_Loss_queue,Av_Q_queue,Max_Q_queue,succes_all,counter_epoch_all,timestep_worker,expl_noise_episode,reward_all,win,all_step,speed,scale_to_goal,Lamda,obs_scale,timestep_episode,logtest,up_speed,smoothness_scale,acceleration_scale,ounoise_decay_step,counter_epoch_collision,collision_queue,counter_epoch_timeout,timeout_queue,step_penalty,W_speed,test,dccs_avg_queue,dccs_std_queue,a_l):
   # writer = SummaryWriter(log_dir=f"runs/1worker_, reward14+nopre") 

    roscoreip =  str(11311)
    gazebo_port =   str(11345)
    # Set the parameters for the implementation
    seed =int(9)  # Random seed number
    eval_freq = 1e4  # After how many steps to perform the evaluation
    max_ep =500  # maximum number of steps per episode
    eval_ep = 10  # number of episodes for evaluation
    max_timesteps = 5e6  # Maximum number of steps to perform
    expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
    expl_decay_steps = (
        30000  # Number of steps over which the initial exploration noise will decay over    default:500000
    )
    expl_min = 0.0  # Exploration noise after the decay in range [0...expl_noise]
    batch_size = 128  # Size of the mini-batch
    discount = 0.9999999  # Discount factor to calculate the discounted future reward (should be close to 1)     #改环境之后，这个也要对应更改，越小的环境这个越小，之前是0.99999
    tau = 0.005  # Soft target update variable (should be close to 0)
    policy_noise = 0.2  # Added noise for exploration
    noise_clip = 0.5  # Maximum clamping values of the noise
    policy_freq = 2  # Frequency of Actor network updates
    buffer_size = 1e6  # Maximum size of the buffer
    file_name = "GAVEL-TD3-"+str(seed)  # name of the file to store the policy
    save_model = True # Weather to save the model or not
    load_model = False # Weather to load a stored model
    random_near_obstacle = False # To take random actions near obstacles or not
    ou_noise = OUNoise(sigma_decay_steps=ounoise_decay_step)
    # Create the network storage folders
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

    # Create the training environment
    environment_dim = 20
    robot_dim = 4
    os.environ['ROS_MASTER_URI'] = f'http://localhost:{roscoreip}'
    os.environ['GAZEBO_MASTER_URI'] = f'http://localhost:{gazebo_port}'
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    launch_file = os.path.join(project_root, "catkin_ws", "src", "multi_robot_scenario", "launch", "TD2_world.launch")
    env = GazeboEnv(launch_file, environment_dim,roscoreip,gazebo_port,t)
    os.environ['ROS_MASTER_URI'] = f'http://localhost:{roscoreip}'
    os.environ['GAZEBO_MASTER_URI'] = f'http://localhost:{gazebo_port}'
    time.sleep(5)
    print(f"Using device: {device}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1
    speed_all=0
    W_SPEED_all=0
    # Create a replay buffer
    replay_buffer = ReplayBuffer(buffer_size, seed)
    if load_model:
        try:
            network.load(file_name, CHECKPOINT_DIR)
        except:
            print(
                "Could not load the stored model parameters, initializing training with random parameters"
            )

    # Create evaluation data store
    evaluations = []

    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    epoch = 1
    
    count_rand_actions = 0
    random_action = []
    episode_reward=0
    episode_timesteps=0
    #choice=False
    # Begin the training loop
    action=[0,0]
    epsilon=0.5
    o_noise=0
    decay_amount=(0.5-0.1)/1500
    noise1=(0,0)
    # if(np.random.uniform(0, 1) > epsilon):
    #         env.choice=True
    # else:
    #     env. choice=False
    env.choice=True 
    env.scale_to_goal=scale_to_goal
    env.obs_scale=obs_scale
    env.smoothness_scale=smoothness_scale
    env.acceleration_scale=acceleration_scale
    env.step_penalty=step_penalty
    initial_eval_offset = 0
    if t == 0:
        run_initial_eval_zero_window(
            network,
            env,
            max_ep,
            max_action,
            success_queue,
            reward_queue,
            reward_all,
            timestep_episode,
        )
        initial_eval_offset = INITIAL_EVAL_EPISODES
        with counter_epoch_all.get_lock():
            counter_epoch_all.value = INITIAL_EVAL_EPISODES - 1
            counter_epoch_success.value = 0
            counter_epoch_collision.value = 0
            counter_epoch_timeout.value = 0
        state = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        env.choice = True
    history_wiggle_action=[]
    wiggle_flag=True
    wiggle_len=5
    initial_lamda = Lamda.value
    lamda_decay_episodes = float(logtest)
    while timestep < max_timesteps:
        
        # On termination of episode
        if done:
                    ou_noise.reset()
                    print("all step:",all_step.value)
                    print("process id:",t)
                    history_wiggle_action.clear()
                    with counter_epoch_all.get_lock():
                        if(env.end_check):
                            counter_epoch_success.value+=1
                            env.end_check=False
                        # if(env.choice):
                        if(episode_timesteps >=max_ep):
                            print("timeout!!!")
                            counter_epoch_timeout.value+=1
                        if(env.collision_check):
                            counter_epoch_collision.value+=1
                            env.collision_check=False
                        counter_epoch_all.value+=1
                        
                        timestep_worker.put((timestep,counter_epoch_all.value))
                        timestep_episode.put((episode_timesteps,counter_epoch_all.value))
                        expl_noise_episode.put((ou_noise.sigma,counter_epoch_all.value))
                        completed_training_episodes = counter_epoch_all.value
                        if initial_eval_offset:
                            completed_training_episodes = (
                                counter_epoch_all.value - initial_eval_offset + 1
                            )
                        Lamda.value = linear_decay_to_zero(
                            initial_lamda,
                            completed_training_episodes,
                            lamda_decay_episodes
                        )
                        print(
                            f"episode={completed_training_episodes}, "
                            f"counter_epoch_all={counter_epoch_all.value}, "
                            f"Lamda={Lamda.value:.6f}"
                        )
                            # if(counter_epoch_all.value>1000):
                            #     expl_noise=0
                        # if(env.end_check):
                        #     if(env.choice):
                        #         counter_epoch_success.value+=1
                        #     env.end_check=False
                        # if(env.choice):
                        #     counter_epoch.value+=1   
                        # if(counter_epoch_all.value<1000):
                        #     env.scale_to_goal=env.scale_to_goal+up_speed/1000
                        #     env.obs_scale-=0.0004 
                        should_log_window = False
                        window_log_epoch = counter_epoch_all.value
                        if initial_eval_offset:
                            if (
                                completed_training_episodes > 0
                                and completed_training_episodes % 50 == 0
                            ):
                                should_log_window = True
                                window_log_epoch = counter_epoch_all.value + 1
                        elif counter_epoch_all.value % 50 == 0:
                            should_log_window = True
                        if(should_log_window):
                                #evaluate(network=network, env=env,epoch=counter_epoch_all.value, eval_episodes=eval_ep,success_queue=success_queue,win=win)
                                succes_all.value=1.0*counter_epoch_success.value/50
                                #writer.add_scalar("Success Rate",  1.0*counter_epoch_success_copy/100,counter_epoch_copy)
                                success_queue.put((1.0*counter_epoch_success.value/50,window_log_epoch))
                                collision_queue.put((1.0*counter_epoch_collision.value/50,window_log_epoch))
                                timeout_queue.put((1.0*counter_epoch_timeout.value/50,window_log_epoch))
                                counter_epoch_collision.value=0
                                counter_epoch_timeout.value=0
                                counter_epoch_success.value=0
                               # counter_epoch_success.value=0
                            
                        if(episode_timesteps!=0 and env.choice):    #这里记录的是真正的自己的奖励，而不是人工市场法的
                            #writer.add_scalar("Rward", episode_reward/episode_timesteps, counter_epoch_copy)
                            reward_queue.put((1.0*episode_reward/episode_timesteps,counter_epoch_all.value))
                            reward_all.put((episode_reward,counter_epoch_all.value))
                            speed.put((speed_all/episode_timesteps,counter_epoch_all.value))    
                            W_speed.put((W_SPEED_all/episode_timesteps,counter_epoch_all.value)) 
                            speed_all=0
                            W_SPEED_all=0
                        #print("oooooooooooooooooooooooooo")
                        if timestep != 0:
                            #print("ttttttttttttttttttttttttttttttttttttttt")
                            network.train(
                                replay_buffer,
                                episode_timesteps,
                                #writer,
                                Critic_Loss_queue,
                                Actor_Loss_queue,
                                dccs_avg_queue,
                                dccs_std_queue,
                                Av_Q_queue,
                                Max_Q_queue,
                                counter_epoch_all.value,
                                batch_size,
                                discount,
                                tau,
                                policy_noise,
                                noise_clip,
                                policy_freq,
                                Lamda.value,
                                a_l
                            )
    
                    if timesteps_since_eval >= eval_freq:
                        #print("Validating")
                        timesteps_since_eval %= eval_freq
                        # evaluations.append(
                        #     evaluate(network=network, env=env,epoch=epoch, eval_episodes=eval_ep)
                        # )
                        print("save!!!!!!!")
                        network.save(file_name, directory=CHECKPOINT_DIR)
                    #    np.save("./results/%s" % (file_name), evaluations)
                        epoch += 1

                    state = env.reset()
                    # 先增加一个新维度，将形状变为 (1, state_dim)
             #       state = np.tile(state, 1) 
                    #env.random_target()
                    done = False

                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1
                    # if(np.random.uniform(0, 1) > epsilon):
                    #     env.choice=True
                    # else:
                    #     env.choice=False 
                    env.choice=True
                    if timestep==0:
                        env.choice=True
                    if(win.value>0.3):
                        epsilon=0
                    else:  
                        epsilon-=decay_amount
                        #epsilon=max(0.1,epsilon)

        
        # add some exploration noise
        if expl_noise > expl_min:
            expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)
            expl_noise=max(0,expl_noise)
         #   ou_noise.decay_sigma()
        odom = env.last_odom
        if odom is None:
            odom = env.wait_for_odom(timeout=30.0, log_label="/odom before APF")
        robot_x = odom.pose.pose.position.x
        robot_y = odom.pose.pose.position.y
        p=env.calculate_attraction_force(robot_x,robot_y,env.goal_x,env.goal_y,650)
        f=env.calculate_total_repulsion_force(robot_x,robot_y,math.sqrt(p[0]**2+p[1]**2))
        z=(f[0]+p[0],f[1]+p[1])
        v = math.sqrt(z[0]**2 + z[1]**2)
        v=min(v,0.5)
        v=max(v,0)
        omega = math.atan2(z[1],z[0])
        #print("angel1:",omega)
        #print("angle2:",env.angle)
        omega=env.angle-omega
        
        if(omega>3.1):
            omega=6.2-omega
            omega=-omega
        if(omega<-3.1):
            omega=omega+6.2
        omega1=omega
        omega=min(omega,1)
        omega=max(omega,-1)
        omega=-omega
        # l=abs(omega)/math.pi*2
        # l=abs(v)-l
        # if(v<0):
        #     v=-l
        # else:
        #      v=l
        if(True):
           # print("mode predict")
            #print("model predict")
            action = network.get_action(np.array(state))
        #    print("model:",action)
          #  noise1=ou_noise.get_noise()
        #     action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        #      -max_action, max_action
        #  )
        #     o_noise=ou_noise.get_noise()
            action = (action + np.random.normal(0, ou_noise.get_noise(), size=action_dim)).clip(
             -max_action, max_action
         )  
        #     action = (action + noise1).clip(
        #     -max_action, max_action
        # )
        # If the robot is facing an obstacle, randomly force it to take a consistent random action.
        # This is done to increase exploration in situations near obstacles.
        # Training can also be performed without it
            if random_near_obstacle:
                if (
                    np.random.uniform(0, 1) > 0.85
                    and min(state[4:-8]) < 0.6
                    and count_rand_actions < 1
                ):
                    count_rand_actions = np.random.randint(8, 15)
                    random_action = np.random.uniform(-1, 1, 2)

                if count_rand_actions > 0:
                    count_rand_actions -= 1
                    action = random_action
                    action[0] = -1
        else:
            #print("artificial")
       
            action[0]=(v*2-1)    #人工市场法需要根据环境来改参数，这里的0.5和1.5是simple环境需要的，还有上面的算引力的32也是，原先是256或者128
            action[1]=omega
        
        # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
        
        a_in = [(action[0]+1)/2, action[1]]
        speed_all+=a_in[0]
        W_SPEED_all=a_in[1]
     #   print(a_in)
        #print(a_in)
        
        
     #   print("artificial:",v*2-1,omega)    
        next_state, reward, done, target = env.step(state,a_in,omega1)
        # 先增加一个新维度，将形状变为 (1, state_dim)
       # next_state = np.concatenate((state, next_state))
      #  next_state=next_state[state_dim:]
        # if(env.choice):
        #     if(len(history_wiggle_action)==wiggle_len ):
        #         for j in range (0,len(history_wiggle_action)-1):
        #             if(history_wiggle_action[j][1]*history_wiggle_action[j+1][1]>0):
        #                 wiggle_flag=False
        #                 break
        #         if(wiggle_flag):
        #             env.choice=False
        #         else:
        #             wiggle_flag=True
        #         history_wiggle_action.clear()
        #     else:
        #         history_wiggle_action.append(a_in)

        done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
        done = 1 if episode_timesteps + 1 == max_ep else int(done)
        episode_reward += reward

        # Save the tuple in replay buffer
        replay_buffer.add(state, action, reward, done_bool, next_state,v*2-1,omega)

        # Update the counters
        state = next_state
        episode_timesteps += 1
        if(env.choice):
            all_step.value+=1
            timestep += 1
        timesteps_since_eval += 1

    # After the training is done, evaluate the network and save it
    # evaluations.append(evaluate(network=network, env=env,epoch=epoch, eval_episodes=eval_ep))
    if save_model:
        network.save("%s" % file_name, directory=FINAL_MODEL_DIR)
    np.save(os.path.join(RESULTS_DIR, "%s.npy" % file_name), evaluations)


def logger_process(k,data_queue, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    while True:
        try:
            item = data_queue.get(timeout=10)
            if item == "END":  # 检测到结束标志，退出循环
                break
            success_rate, epoch = item
            if(k==0):
                writer.add_scalar("Reward", success_rate, epoch)
      #          print("11111111111111111111111111111")
            if (k==1):
                writer.add_scalar("Success Rate", success_rate, epoch)
          #      print("333333333333333333333333333333")
            if(k==2):
                writer.add_scalar("Critic-Loss", success_rate, epoch)
          #      print("4444444444444444444444444444")
            if(k==3):
          #      print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz:",epoch)
                writer.add_scalar("Actor-Loss", success_rate, epoch)
          #      print("5555555555555555555555555555555555") 
            if(k==4):
                writer.add_scalar("Av_Q", success_rate, epoch)
          #      print("666666666666666666666666666666666")
            if(k==5):
                writer.add_scalar("Max-Q", success_rate, epoch)
            #    print("777777777777777777777777777777777")   
            if(k==6):
                writer.add_scalar("time_steps", success_rate, epoch)
            #    print("88888888888888888888888888888888")   
            if(k==7):
                writer.add_scalar("expl_noise", success_rate, epoch)
          #      print("9999999999999999999999999999999999")   
            if(k==8):
                writer.add_scalar("Episode Rward", success_rate, epoch)
           #     print("jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj")   
            if(k==9):
                writer.add_scalar("episode step",success_rate,epoch)
            if(k==10):
                writer.add_scalar("linear speed",success_rate,epoch)
            if(k==11):
                writer.add_scalar("collision",success_rate,epoch)
            if(k==12):
                writer.add_scalar("timeout",success_rate,epoch)
            if(k==13):
                writer.add_scalar("angular_speed",success_rate,epoch)
            if(k==14):
                writer.add_scalar("dccs_avg_queue",success_rate,epoch)
            if(k==15):
                writer.add_scalar("dccs_std_queue",success_rate,epoch)
        except queue.Empty:  # 捕获 queue.Empty 异常
            continue
    writer.close()
    
if __name__ == '__main__':
    mp.set_start_method('spawn')

    scale_to_goal = 0.0
    obs_scale=40.0
    logtest=100.0
    test=333.0
    up_speed=740.0
    smoothness_scale=0.0
    acceleration_scale=0.0
    ounoise_decay_step=500.0
    step_penalty=-0.0008
    a_l=46.0
    environment_dim = 20
    robot_dim = 4
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1
    MasterNode = TD3(state_dim, action_dim, max_action) #A
    MasterNode.share_memory() #B
    counter_epoch=mp.Value('i',0)
    succes_all=mp.Value('d',0.0)
    win=mp.Value('d',0.0)
    counter_epoch_all=mp.Value('i',0)
    counter_epoch_success=mp.Value('i',0)
    all_step=mp.Value('i',0)
    Lamda = mp.Value('d', 0.0)
    counter_epoch_timeout=mp.Value('i',0)
    counter_epoch_collision=mp.Value('i',0)
    Lamda.value=1
    if len(sys.argv) not in (1, 11):
        raise SystemExit(
            "Expected either no arguments or 10 positional hyperparameters; "
            "use scripts/train.sh for the verified defaults."
        )
    if len(sys.argv) == 11:
        # 获取传递的参数并转换为浮点数
        scale_to_goal = float(sys.argv[1])  # 使用 float() 获取浮点数值
        obs_scale = float(sys.argv[2])
        logtest = float(sys.argv[3])
        test = float(sys.argv[4])
        up_speed=float(sys.argv[5])
        smoothness_scale=float(sys.argv[6])
        acceleration_scale=float(sys.argv[7])
        ounoise_decay_step=float(sys.argv[8])
        step_penalty=float(sys.argv[9])
        a_l=float(sys.argv[10])
       # decay_steps = float(sys.argv[3])  # decay_steps 也改为浮点数

    #print("dsadsadsa:",Lamda.value,decay_steps)
    run_name = f"reward+{scale_to_goal}+{obs_scale}+{logtest}+{test}+{up_speed}+{smoothness_scale}+{acceleration_scale}+{ounoise_decay_step}+{step_penalty}+{a_l}"
    logdir = os.path.join(TENSORBOARD_DIR, run_name)
    processes=[]
    params={'n_workers':1,
            'n_loggers':2,
            }
    success_queue = mp.Queue()
    reward_queue = mp.Queue()
    Critic_Loss_queue=mp.Queue()
    Actor_Loss_queue=mp.Queue()
    Max_Q_queue=mp.Queue()
    Av_Q_queue=mp.Queue()
    timeout_queue=mp.Queue()
    collision_queue=mp.Queue()
    loss_queue = mp.Queue()
    timestep_worker = mp.Queue()
    timestep_episode = mp.Queue()
    expl_noise_episode = mp.Queue()
    reward_all=mp.Queue()
    speed=mp.Queue()  
    W_speed=mp.Queue()  
    dccs_avg_queue=mp.Queue()
    dccs_std_queue=mp.Queue()
    logger = mp.Process(target=logger_process, args=(0,reward_queue, logdir))
    logger.start()
    processes.append(logger)
    
    logger1 = mp.Process(target=logger_process, args=(1,success_queue, logdir))
    logger1.start()
    processes.append(logger1)   

    logger2 = mp.Process(target=logger_process, args=(2,Critic_Loss_queue, logdir))
    logger2.start()
    processes.append(logger2) 
    
    logger3 = mp.Process(target=logger_process, args=(3,Actor_Loss_queue, logdir))
    logger3.start()
    processes.append(logger3) 
    
    logger4 = mp.Process(target=logger_process, args=(4,Av_Q_queue, logdir))
    logger4.start()
    processes.append(logger4) 
    
    logger5 = mp.Process(target=logger_process, args=(5,Max_Q_queue, logdir))
    logger5.start()
    processes.append(logger5) 
    
    logger6 = mp.Process(target=logger_process, args=(6,timestep_worker, logdir))
    logger6.start()
    processes.append(logger6)
    
    logger7 = mp.Process(target=logger_process, args=(7,expl_noise_episode, logdir))
    logger7.start()
    processes.append(logger7)
    
    logger8 = mp.Process(target=logger_process, args=(8,reward_all, logdir))
    logger8.start()
    processes.append(logger8)

    logger9 = mp.Process(target=logger_process, args=(9,timestep_episode, logdir))
    logger9.start()
    processes.append(logger9)

    logger10 = mp.Process(target=logger_process, args=(10,speed, logdir))
    logger10.start()
    processes.append(logger10)

    logger11 = mp.Process(target=logger_process, args=(11,collision_queue, logdir))
    logger11.start()
    processes.append(logger11)
    
    logger12 = mp.Process(target=logger_process, args=(12,timeout_queue, logdir))
    logger12.start()
    processes.append(logger12)
    
    logger13 = mp.Process(target=logger_process, args=(13,W_speed, logdir))
    logger13.start()
    processes.append(logger13)

    logger14 = mp.Process(target=logger_process, args=(14,dccs_avg_queue, logdir))
    logger14.start()
    processes.append(logger14)

    logger15 = mp.Process(target=logger_process, args=(15,dccs_std_queue, logdir))
    logger15.start()
    processes.append(logger15)
    for i in range(params["n_workers"]):
            p = mp.Process(target=worker, args=(i, MasterNode, counter_epoch, counter_epoch_success, params, success_queue, reward_queue, Critic_Loss_queue, Actor_Loss_queue, Av_Q_queue, Max_Q_queue, succes_all, counter_epoch_all, timestep_worker, expl_noise_episode, reward_all, win, all_step,speed,scale_to_goal,Lamda,obs_scale,timestep_episode,logtest,up_speed,smoothness_scale,acceleration_scale,ounoise_decay_step,counter_epoch_collision,collision_queue,counter_epoch_timeout,timeout_queue,step_penalty,W_speed,test,dccs_avg_queue,dccs_std_queue,a_l))
            p.start()
            processes.append(p)
    for p in processes:
            p.join()
    for p in processes:
            p.terminate()
