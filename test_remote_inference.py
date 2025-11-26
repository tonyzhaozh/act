#!/usr/bin/env python3
"""
ACT Policy 远程推理客户端示例

演示如何连接到 ACT Policy 服务器并获取动作
"""

import numpy as np
import time
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'web_policy'))

from web_policy import WebSocketClientPolicy


def create_dummy_observation(metadata):
    """
    根据元数据创建虚拟观测数据
    
    Args:
        metadata: 服务器返回的元数据
    
    Returns:
        obs: 观测数据字典
    """
    obs = {}
    
    # 创建 qpos（关节位置）
    state_dim = metadata['state_dim']
    obs['qpos'] = np.random.randn(state_dim).astype(np.float32)
    print(f"   qpos: shape={obs['qpos'].shape}, dtype={obs['qpos'].dtype}")
    
    # 创建图像数据（假设 640x480 RGB）
    camera_names = metadata['camera_names']
    for cam_name in camera_names:
        obs[cam_name] = np.random.randint(
            0, 255, 
            size=(480, 640, 3), 
            dtype=np.uint8
        )
        print(f"   {cam_name}: shape={obs[cam_name].shape}, dtype={obs[cam_name].dtype}")
    
    return obs


def main():
    print("=" * 60)
    print("ACT Policy 远程推理客户端")
    print("=" * 60)
    
    # 连接到服务器
    print("\n🔌 连接到服务器 localhost:8000...")
    client = WebSocketClientPolicy(
        host="localhost",
        port=8000
    )
    
    # 获取服务器元数据
    print("\n📊 服务器元数据:")
    metadata = client.get_server_metadata()
    for key, value in metadata.items():
        print(f"   {key}: {value}")
    
    num_queries = metadata['num_queries']
    print(f"\n📸 创建虚拟观测数据:")
    obs = create_dummy_observation(metadata)
    
    # 推理测试
    print(f"\n🚀 执行推理测试...")
    num_tests = 5
    
    for i in range(num_tests):
        print(f"\n--- 推理 {i+1}/{num_tests} ---")
        
        start_time = time.time()
        result = client.infer(obs)
        inference_time = time.time() - start_time
        
        actions = result['actions']
        server_timing = result.get('server_timing', {})
        
        print(f"✅ 推理成功!")
        print(f"   动作 shape: {actions.shape}")
        print(f"   动作序列长度 (chunk_size): {actions.shape[0]}")
        print(f"   每个动作维度: {actions.shape[1]}")
        print(f"   动作范围: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"   服务器推理耗时: {server_timing.get('infer_ms', 0):.2f} ms")
        print(f"   客户端总耗时: {inference_time*1000:.2f} ms")
        
        # 模拟控制频率
        time.sleep(0.1)
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("\n💡 使用说明:")
    print(f"   - ACT 策略每次推理返回 {num_queries} 步动作序列")
    print(f"   - 可以逐步执行这些动作，也可以根据 query_frequency 重新推理")
    print(f"   - 如果使用 temporal aggregation，需要在客户端实现")
    print("=" * 60)


if __name__ == '__main__':
    main()
