#!/usr/bin/env python3
"""
ACT Policy 远程推理示例 - 支持 Temporal Aggregation

演示如何使用 temporal aggregation 策略执行动作
这是 ACT 论文中提出的重要技术，可以提高动作的平滑性和稳定性
"""

import numpy as np
import time
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'web_policy'))

from web_policy import WebSocketClientPolicy


class ACTRemoteInferenceClient:
    """
    ACT Policy 远程推理客户端（支持 Temporal Aggregation）
    
    Temporal Aggregation 原理：
    - 每次推理返回 num_queries 步的动作序列
    - 对于第 t 步，收集所有预测过该步的动作
    - 使用指数加权平均融合这些动作预测
    - 越近期的预测权重越大
    """
    
    def __init__(self, host='localhost', port=8000, temporal_agg=True, k=0.01):
        """
        Args:
            host: 服务器地址
            port: 服务器端口
            temporal_agg: 是否使用 temporal aggregation
            k: 指数衰减系数（越大表示越重视近期预测）
        """
        self.client = WebSocketClientPolicy(host=host, port=port)
        self.metadata = self.client.get_server_metadata()
        
        self.num_queries = self.metadata['num_queries']
        self.action_dim = self.metadata['action_dim']
        self.temporal_agg = temporal_agg
        self.k = k
        
        # Temporal aggregation 缓存
        # all_time_actions[t, t':t'+num_queries] 存储在时刻 t 预测的动作序列
        self.max_timesteps = 1000  # 最大时间步数
        self.all_time_actions = np.zeros(
            (self.max_timesteps, self.max_timesteps + self.num_queries, self.action_dim),
            dtype=np.float32
        )
        
        self.query_frequency = 1 if temporal_agg else self.num_queries
        
        print(f"📊 ACT 远程推理客户端初始化:")
        print(f"   num_queries (chunk_size): {self.num_queries}")
        print(f"   action_dim: {self.action_dim}")
        print(f"   temporal_aggregation: {temporal_agg}")
        print(f"   query_frequency: {self.query_frequency}")
        if temporal_agg:
            print(f"   decay_factor (k): {k}")
    
    def reset(self):
        """重置 temporal aggregation 缓存"""
        self.all_time_actions.fill(0)
    
    def get_action(self, obs, t):
        """
        获取第 t 步的动作
        
        Args:
            obs: 观测数据字典
            t: 当前时间步
        
        Returns:
            action: 单步动作 (action_dim,)
        """
        # 是否需要重新查询 policy
        if t % self.query_frequency == 0:
            # 推理获取整个动作序列
            result = self.client.infer(obs)
            all_actions = result['actions']  # (num_queries, action_dim)
            
            if self.temporal_agg:
                # 存储这次预测的所有动作
                # 在时刻 t 预测的动作对应 t, t+1, ..., t+num_queries-1
                self.all_time_actions[t, t:t+self.num_queries] = all_actions
        
        if self.temporal_agg:
            # 使用 temporal aggregation 融合动作
            # 收集所有预测过第 t 步的动作
            actions_for_curr_step = self.all_time_actions[:, t]  # (max_timesteps, action_dim)
            
            # 找出哪些预测是有效的（非零）
            actions_populated = np.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]  # (n_valid, action_dim)
            
            # 计算指数加权平均
            # 权重随时间衰减：w_i = exp(-k * i)，i 是距离当前的时间差
            n_valid = len(actions_for_curr_step)
            exp_weights = np.exp(-self.k * np.arange(n_valid))
            exp_weights = exp_weights / exp_weights.sum()  # 归一化
            
            # 加权平均
            action = np.sum(actions_for_curr_step * exp_weights[:, np.newaxis], axis=0)
        else:
            # 不使用 temporal aggregation，直接使用当前预测
            result = self.client.infer(obs)
            all_actions = result['actions']
            action = all_actions[t % self.query_frequency]
        
        return action


def create_dummy_observation(metadata):
    """创建虚拟观测数据"""
    obs = {}
    
    # qpos
    state_dim = metadata['state_dim']
    obs['qpos'] = np.random.randn(state_dim).astype(np.float32)
    
    # 图像
    camera_names = metadata['camera_names']
    for cam_name in camera_names:
        obs[cam_name] = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
    
    return obs


def main():
    print("=" * 70)
    print("ACT Policy 远程推理 - Temporal Aggregation 示例")
    print("=" * 70)
    
    # 创建客户端（启用 temporal aggregation）
    client = ACTRemoteInferenceClient(
        host='localhost',
        port=8000,
        temporal_agg=True,
        k=0.01
    )
    
    # 模拟一个 episode
    max_timesteps = 50
    print(f"\n🎬 开始模拟 episode (max_timesteps={max_timesteps})")
    
    client.reset()
    
    actions_list = []
    inference_times = []
    
    for t in range(max_timesteps):
        # 创建观测（实际使用时替换为真实观测）
        obs = create_dummy_observation(client.metadata)
        
        # 获取动作
        start_time = time.time()
        action = client.get_action(obs, t)
        inference_time = time.time() - start_time
        
        actions_list.append(action)
        inference_times.append(inference_time)
        
        if t % 10 == 0:
            print(f"   步骤 {t}: 动作 shape={action.shape}, "
                  f"范围=[{action.min():.3f}, {action.max():.3f}], "
                  f"耗时={inference_time*1000:.2f}ms")
    
    print(f"\n✅ Episode 完成!")
    print(f"   总步数: {max_timesteps}")
    print(f"   平均推理耗时: {np.mean(inference_times)*1000:.2f} ms")
    print(f"   推理频率: 每 {client.query_frequency} 步查询一次")
    
    # 分析动作平滑度
    actions_array = np.array(actions_list)  # (max_timesteps, action_dim)
    action_diff = np.diff(actions_array, axis=0)  # (max_timesteps-1, action_dim)
    action_smoothness = np.mean(np.abs(action_diff))
    
    print(f"\n📊 动作统计:")
    print(f"   动作维度: {actions_array.shape[1]}")
    print(f"   动作平均变化量: {action_smoothness:.4f}")
    print(f"   （越小表示动作越平滑）")
    
    print("\n" + "=" * 70)
    print("💡 Temporal Aggregation 的优势:")
    print("   1. 动作更平滑，减少抖动")
    print("   2. 利用历史预测信息，提高鲁棒性")
    print("   3. 在线推理时可以每步都查询，提高响应速度")
    print("=" * 70)


if __name__ == '__main__':
    main()
