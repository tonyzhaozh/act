#!/usr/bin/env python3
"""
ACT Policy 远程推理服务器
使用 web_policy 提供 WebSocket 推理服务

Usage:
    python serve_act_policy.py -i <checkpoint_path> -p 8000
"""

import sys
import os
import click
import torch
import pickle
import numpy as np
from einops import rearrange

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'web_policy'))

from web_policy import BasePolicy, WebSocketPolicyServer
from policy import ACTPolicy
from einops import rearrange


class ACTPolicyWrapper(BasePolicy):
    """
    包装 ACT Policy 为 BasePolicy 接口
    处理所有归一化、反归一化和数据转换
    """
    
    def __init__(self, ckpt_path: str, device: str = 'cuda'):
        """
        Args:
            ckpt_path: checkpoint 文件路径
            device: 'cuda' 或 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 从 checkpoint 路径推断 ckpt_dir 和 stats 路径
        ckpt_dir = os.path.dirname(ckpt_path)
        stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
        
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"找不到 dataset_stats.pkl，应该在: {stats_path}\n"
                f"请确保 checkpoint 目录中有 dataset_stats.pkl 文件"
            )
        
        # 加载统计信息（用于归一化）
        print(f"🔄 加载统计信息: {stats_path}")
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        
        print(f"📊 数据集统计信息:")
        print(f"   qpos_mean shape: {self.stats['qpos_mean'].shape}")
        print(f"   qpos_std shape: {self.stats['qpos_std'].shape}")
        print(f"   action_mean shape: {self.stats['action_mean'].shape}")
        print(f"   action_std shape: {self.stats['action_std'].shape}")
        
        # 加载 checkpoint
        print(f"🔄 加载 checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        # 从 state_dict 推断模型配置
        # 注意：ACT checkpoint 只保存了权重，需要从权重推断配置
        self.state_dim = self.stats['qpos_mean'].shape[0]
        
        # 尝试从 checkpoint 文件名或配置推断参数
        # 默认配置（可能需要根据实际情况调整）
        policy_config = {
            'lr': 1e-5,
            'num_queries': 100,  # chunk_size，默认值
            'kl_weight': 10,
            'hidden_dim': 512,
            'dim_feedforward': 3200,
            'lr_backbone': 1e-5,
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': ['top'],  # 默认，需要根据实际情况修改
        }
        
        print(f"📊 模型配置 (默认值，可能需要调整):")
        print(f"   num_queries (chunk_size): {policy_config['num_queries']}")
        print(f"   state_dim: {self.state_dim}")
        
        # 创建 policy
        self.policy = ACTPolicy(policy_config)
        self.policy.load_state_dict(state_dict)
        self.policy = self.policy.eval().to(self.device)
        
        # 注意：图像归一化在 policy.__call__() 内部自动完成，无需在此处理
        
        self.num_queries = policy_config['num_queries']
        self.camera_names = policy_config['camera_names']
        
        print(f"✅ Policy 加载成功")
        print(f"   设备: {self.device}")
        print(f"   chunk_size (num_queries): {self.num_queries}")
    
    def reset(self):
        """重置 policy 状态"""
        # ACT Policy 没有内部状态需要重置
        pass
    
    def infer(self, obs: dict) -> dict:
        """
        推理方法
        
        Args:
            obs: 观测数据字典，格式要求：
                {
                    'qpos': np.ndarray,  # shape: (state_dim,), 当前关节位置
                    'camera_0' (or other camera names): np.ndarray,  # shape: (H, W, C), RGB 图像
                    'camera_1': ...,  # 可选，多个相机
                }
        
        Returns:
            结果字典:
                {
                    'actions': np.ndarray,  # shape: (num_queries, action_dim), 动作序列
                }
        """
        # ========== 1. 处理 qpos（关节位置）归一化 ==========
        qpos_numpy = obs['qpos']
        # 归一化: (qpos - mean) / std
        qpos_normalized = (qpos_numpy - self.stats['qpos_mean']) / self.stats['qpos_std']
        qpos = torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)  # (1, state_dim)
        
        # ========== 2. 处理图像 ==========
        # 重要：必须先组装成 (1, num_cameras, C, H, W) 格式，再统一归一化
        # 这样才能与训练时保持一致（训练时在 policy.__call__ 中对整个 tensor 归一化）
        curr_images = []
        for cam_name in self.camera_names:
            if cam_name not in obs:
                raise ValueError(f"缺少相机 '{cam_name}' 的图像数据")
            
            curr_image = obs[cam_name]  # (H, W, C), RGB, uint8 或 float32
            
            # 转换为 float32 [0, 1]
            if curr_image.dtype == np.uint8:
                curr_image = curr_image.astype(np.float32) / 255.0
            
            # HWC -> CHW
            curr_image = rearrange(curr_image, 'h w c -> c h w')
            curr_images.append(curr_image)
        
        # 堆叠所有相机图像: (num_cameras, C, H, W)
        curr_image = np.stack(curr_images, axis=0)
        # 转换为 tensor 并添加 batch 维度: (1, num_cameras, C, H, W)
        curr_image = torch.from_numpy(curr_image).float().to(self.device).unsqueeze(0)
        
        # ImageNet 归一化（对整个 tensor 进行，与训练时一致）
        # 注意：这会在 policy.__call__() 中再次归一化，但为了与训练完全一致，我们在这里也不做
        # 实际上，训练时归一化在 policy 内部，所以这里不应该归一化
        
        # ========== 3. 推理 ==========
        with torch.no_grad():
            # policy 内部会做 ImageNet 归一化
            # ACT 返回整个动作序列 (1, num_queries, action_dim)
            actions = self.policy(qpos, curr_image)
        
        # ========== 4. 反归一化动作 ==========
        # unnormalize: action * std + mean
        actions_numpy = actions[0].cpu().numpy()  # (num_queries, action_dim)
        actions_unnormalized = actions_numpy * self.stats['action_std'] + self.stats['action_mean']
        
        # 返回结果
        return {
            'actions': actions_unnormalized,
        }
    
    @property
    def metadata(self) -> dict:
        """返回 policy 元数据"""
        return {
            'model': 'ACT',
            'num_queries': self.num_queries,
            'state_dim': self.state_dim,
            'action_dim': self.stats['action_mean'].shape[0],
            'device': str(self.device),
            'camera_names': self.camera_names,
        }


@click.command()
@click.option('--input', '-i', required=True, help='Checkpoint 文件路径 (policy_best.ckpt)')
@click.option('--port', '-p', default=8000, type=int, help='服务器端口')
@click.option('--host', '-h', default='0.0.0.0', help='服务器地址')
@click.option('--device', '-d', default='cuda', help='设备: cuda 或 cpu')
@click.option('--camera_names', '-c', default='top', help='相机名称，逗号分隔，例如: top,front')
@click.option('--num_queries', '-q', default=100, type=int, help='Chunk size (num_queries)')
def main(input, port, host, device, camera_names, num_queries):
    """启动 ACT Policy 远程推理服务器"""
    
    print("=" * 60)
    print("ACT Policy 远程推理服务器")
    print("=" * 60)
    
    # 解析相机名称
    camera_names_list = [name.strip() for name in camera_names.split(',')]
    
    # 创建 policy wrapper
    policy = ACTPolicyWrapper(
        ckpt_path=input,
        device=device
    )
    
    # 更新相机配置
    policy.camera_names = camera_names_list
    policy.num_queries = num_queries
    
    print(f"\n📊 Policy 元数据:")
    for key, value in policy.metadata.items():
        print(f"   {key}: {value}")
    
    # 预热推理
    print(f"\n🔥 预热推理...")
    try:
        # 创建虚拟观测数据
        dummy_obs = {
            'qpos': np.random.randn(policy.state_dim).astype(np.float32),
        }
        
        # 添加虚拟图像（假设 640x480 RGB）
        for cam_name in policy.camera_names:
            dummy_obs[cam_name] = np.random.randint(
                0, 255, size=(480, 640, 3), dtype=np.uint8
            )
        
        result = policy.infer(dummy_obs)
        print(f"✅ 预热成功！动作 shape: {result['actions'].shape}")
    except Exception as e:
        print(f"⚠️ 预热失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 启动服务器
    print(f"\n🚀 启动 WebSocket 服务器...")
    print(f"   地址: {host}:{port}")
    print(f"   健康检查: http://localhost:{port}/healthz")
    print(f"\n💡 提示:")
    print(f"   - 相机列表: {camera_names_list}")
    print(f"   - 使用 WebSocketClientPolicy 连接此服务器")
    print(f"   - 按 Ctrl+C 停止服务器")
    print("=" * 60)
    
    server = WebSocketPolicyServer(
        policy=policy,
        host=host,
        port=port
    )
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止")


if __name__ == '__main__':
    main()
