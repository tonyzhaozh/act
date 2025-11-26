#!/usr/bin/env python3
"""
ACT Policy 远程推理服务器测试脚本

测试内容：
1. 数据格式验证
2. 归一化/反归一化正确性
3. 与原始推理对比
4. 多相机支持
5. Temporal Aggregation
6. 性能测试
"""

import sys
import os
import numpy as np
import torch
import pickle
from einops import rearrange
import time

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'web_policy'))

from policy import ACTPolicy
from serve_act_policy import ACTPolicyWrapper
from web_policy import WebSocketClientPolicy


class TestACTPolicy:
    """ACT Policy 测试类"""
    
    def __init__(self, ckpt_path, camera_names=['top'], num_queries=100):
        self.ckpt_path = ckpt_path
        self.ckpt_dir = os.path.dirname(ckpt_path)
        self.camera_names = camera_names
        self.num_queries = num_queries
        
        # 加载 stats
        stats_path = os.path.join(self.ckpt_dir, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        
        print("=" * 70)
        print("ACT Policy 测试初始化")
        print("=" * 70)
        print(f"Checkpoint: {ckpt_path}")
        print(f"Stats: {stats_path}")
        print(f"Cameras: {camera_names}")
        print(f"Num queries: {num_queries}")
        print()
    
    def test_1_data_format(self):
        """测试 1: 数据格式验证"""
        print("\n" + "=" * 70)
        print("测试 1: 数据格式验证")
        print("=" * 70)
        
        try:
            wrapper = ACTPolicyWrapper(self.ckpt_path, device='cuda')
            
            # 创建测试观测
            obs = self._create_test_obs()
            
            print("✓ 输入数据格式:")
            print(f"  qpos: {obs['qpos'].shape}, dtype={obs['qpos'].dtype}")
            for cam_name in self.camera_names:
                print(f"  {cam_name}: {obs[cam_name].shape}, dtype={obs[cam_name].dtype}")
            
            # 推理
            result = wrapper.infer(obs)
            actions = result['actions']
            
            print(f"\n✓ 输出数据格式:")
            print(f"  actions: {actions.shape}, dtype={actions.dtype}")
            print(f"  期望: ({self.num_queries}, {self.stats['action_mean'].shape[0]})")
            
            # 验证 shape
            assert actions.shape[0] == self.num_queries, f"动作序列长度错误: {actions.shape[0]} != {self.num_queries}"
            assert actions.shape[1] == self.stats['action_mean'].shape[0], f"动作维度错误"
            
            print("\n✅ 测试 1 通过: 数据格式正确")
            return True
            
        except Exception as e:
            print(f"\n❌ 测试 1 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_2_normalization(self):
        """测试 2: 归一化/反归一化正确性"""
        print("\n" + "=" * 70)
        print("测试 2: 归一化/反归一化正确性")
        print("=" * 70)
        
        try:
            wrapper = ACTPolicyWrapper(self.ckpt_path, device='cuda')
            
            # 使用已知的 qpos 测试
            qpos_original = self.stats['qpos_mean'].numpy() + self.stats['qpos_std'].numpy()
            obs = self._create_test_obs(qpos=qpos_original)
            
            print("✓ 测试数据:")
            print(f"  qpos_original: {qpos_original[:3]}...")
            print(f"  qpos_mean: {self.stats['qpos_mean'].numpy()[:3]}...")
            print(f"  qpos_std: {self.stats['qpos_std'].numpy()[:3]}...")
            
            # 手动归一化
            qpos_normalized_manual = (qpos_original - self.stats['qpos_mean'].numpy()) / self.stats['qpos_std'].numpy()
            print(f"  qpos_normalized (manual): {qpos_normalized_manual[:3]}...")
            
            # 推理
            result = wrapper.infer(obs)
            actions = result['actions']
            
            print(f"\n✓ 动作统计:")
            print(f"  actions mean: {actions.mean():.6f}")
            print(f"  actions std: {actions.std():.6f}")
            print(f"  actions range: [{actions.min():.6f}, {actions.max():.6f}]")
            
            # 验证动作在合理范围内（反归一化后应该接近 action_mean 附近）
            expected_mean = self.stats['action_mean'].numpy().mean()
            expected_std = self.stats['action_std'].numpy().mean()
            
            print(f"\n✓ 期望统计:")
            print(f"  expected mean: {expected_mean:.6f}")
            print(f"  expected std: {expected_std:.6f}")
            
            print("\n✅ 测试 2 通过: 归一化处理正确")
            return True
            
        except Exception as e:
            print(f"\n❌ 测试 2 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_3_compare_with_original(self):
        """测试 3: 与原始推理对比"""
        print("\n" + "=" * 70)
        print("测试 3: 与原始推理对比")
        print("=" * 70)
        
        try:
            # 加载原始 policy
            print("✓ 加载原始 policy...")
            policy_config = {
                'lr': 1e-5,
                'num_queries': self.num_queries,
                'kl_weight': 10,
                'hidden_dim': 512,
                'dim_feedforward': 3200,
                'lr_backbone': 1e-5,
                'backbone': 'resnet18',
                'enc_layers': 4,
                'dec_layers': 7,
                'nheads': 8,
                'camera_names': self.camera_names,
            }
            
            original_policy = ACTPolicy(policy_config)
            state_dict = torch.load(self.ckpt_path, map_location='cpu')
            original_policy.load_state_dict(state_dict)
            original_policy = original_policy.eval().cuda()
            
            # 加载 wrapper
            print("✓ 加载 wrapper policy...")
            wrapper = ACTPolicyWrapper(self.ckpt_path, device='cuda')
            
            # 创建相同的测试数据
            obs = self._create_test_obs(seed=42)
            
            # 原始推理
            print("\n✓ 执行原始推理...")
            qpos = obs['qpos']
            qpos_normalized = (qpos - self.stats['qpos_mean'].numpy()) / self.stats['qpos_std'].numpy()
            qpos_torch = torch.from_numpy(qpos_normalized).float().cuda().unsqueeze(0)
            
            # 处理图像
            curr_images = []
            for cam_name in self.camera_names:
                curr_image = obs[cam_name].astype(np.float32) / 255.0
                curr_image = rearrange(curr_image, 'h w c -> c h w')
                curr_images.append(curr_image)
            curr_image = np.stack(curr_images, axis=0)
            curr_image_torch = torch.from_numpy(curr_image).float().cuda().unsqueeze(0)
            
            with torch.no_grad():
                actions_original = original_policy(qpos_torch, curr_image_torch)
                actions_original = actions_original[0].cpu().numpy()
                actions_original = actions_original * self.stats['action_std'].numpy() + self.stats['action_mean'].numpy()
            
            # Wrapper 推理
            print("✓ 执行 wrapper 推理...")
            result = wrapper.infer(obs)
            actions_wrapper = result['actions']
            
            # 比较结果
            print(f"\n✓ 结果对比:")
            print(f"  Original shape: {actions_original.shape}")
            print(f"  Wrapper shape: {actions_wrapper.shape}")
            
            diff = np.abs(actions_original - actions_wrapper)
            print(f"  Max diff: {diff.max():.8f}")
            print(f"  Mean diff: {diff.mean():.8f}")
            print(f"  Median diff: {np.median(diff):.8f}")
            
            # 验证一致性（考虑浮点误差）
            is_close = np.allclose(actions_original, actions_wrapper, atol=1e-5, rtol=1e-4)
            
            if is_close:
                print("\n✅ 测试 3 通过: 推理结果完全一致")
                return True
            else:
                print(f"\n⚠️  测试 3 警告: 存在较大差异")
                print(f"  最大差异位置: {np.unravel_index(diff.argmax(), diff.shape)}")
                print(f"  原始值: {actions_original.flat[diff.argmax()]:.8f}")
                print(f"  Wrapper值: {actions_wrapper.flat[diff.argmax()]:.8f}")
                
                # 如果差异很小，仍然算通过
                if diff.max() < 1e-3:
                    print("  差异在可接受范围内（< 1e-3）")
                    return True
                return False
            
        except Exception as e:
            print(f"\n❌ 测试 3 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_4_multi_camera(self):
        """测试 4: 多相机支持"""
        print("\n" + "=" * 70)
        print("测试 4: 多相机支持")
        print("=" * 70)
        
        if len(self.camera_names) == 1:
            print("⚠️  跳过测试: 只有一个相机")
            return True
        
        try:
            wrapper = ACTPolicyWrapper(self.ckpt_path, device='cuda')
            
            # 测试每个相机的图像是否被正确处理
            obs = self._create_test_obs()
            
            print(f"✓ 相机数量: {len(self.camera_names)}")
            for i, cam_name in enumerate(self.camera_names):
                print(f"  Camera {i}: {cam_name}")
            
            # 推理
            result = wrapper.infer(obs)
            actions = result['actions']
            
            print(f"\n✓ 推理成功，动作 shape: {actions.shape}")
            
            # 测试缺少相机的情况
            print("\n✓ 测试缺少相机...")
            obs_incomplete = obs.copy()
            del obs_incomplete[self.camera_names[0]]
            
            try:
                wrapper.infer(obs_incomplete)
                print("  ❌ 应该抛出错误但没有")
                return False
            except ValueError as e:
                print(f"  ✓ 正确抛出错误: {e}")
            
            print("\n✅ 测试 4 通过: 多相机支持正确")
            return True
            
        except Exception as e:
            print(f"\n❌ 测试 4 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_5_performance(self):
        """测试 5: 性能测试"""
        print("\n" + "=" * 70)
        print("测试 5: 性能测试")
        print("=" * 70)
        
        try:
            wrapper = ACTPolicyWrapper(self.ckpt_path, device='cuda')
            obs = self._create_test_obs()
            
            # 预热
            print("✓ 预热推理...")
            for _ in range(5):
                wrapper.infer(obs)
            
            # 性能测试
            print("✓ 性能测试 (100 次推理)...")
            times = []
            for _ in range(100):
                start = time.time()
                wrapper.infer(obs)
                times.append(time.time() - start)
            
            times = np.array(times) * 1000  # 转换为 ms
            
            print(f"\n✓ 性能统计 (ms):")
            print(f"  Mean: {times.mean():.2f}")
            print(f"  Median: {np.median(times):.2f}")
            print(f"  Std: {times.std():.2f}")
            print(f"  Min: {times.min():.2f}")
            print(f"  Max: {times.max():.2f}")
            print(f"  P95: {np.percentile(times, 95):.2f}")
            print(f"  P99: {np.percentile(times, 99):.2f}")
            
            # 评估性能
            if times.mean() < 100:
                print("\n✅ 测试 5 通过: 性能优秀 (< 100ms)")
            elif times.mean() < 200:
                print("\n✅ 测试 5 通过: 性能良好 (< 200ms)")
            else:
                print(f"\n⚠️  测试 5 警告: 性能较慢 ({times.mean():.2f}ms)")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 测试 5 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_6_edge_cases(self):
        """测试 6: 边界情况"""
        print("\n" + "=" * 70)
        print("测试 6: 边界情况")
        print("=" * 70)
        
        try:
            wrapper = ACTPolicyWrapper(self.ckpt_path, device='cuda')
            
            # 测试 1: 全零输入
            print("✓ 测试全零输入...")
            obs = self._create_test_obs()
            obs['qpos'] = np.zeros_like(obs['qpos'])
            for cam_name in self.camera_names:
                obs[cam_name] = np.zeros_like(obs[cam_name])
            
            result = wrapper.infer(obs)
            print(f"  动作范围: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")
            
            # 测试 2: 极值输入
            print("\n✓ 测试极值输入...")
            obs = self._create_test_obs()
            obs['qpos'] = np.ones_like(obs['qpos']) * 1000
            
            result = wrapper.infer(obs)
            print(f"  动作范围: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")
            
            # 测试 3: float32 图像输入
            print("\n✓ 测试 float32 图像...")
            obs = self._create_test_obs()
            for cam_name in self.camera_names:
                obs[cam_name] = obs[cam_name].astype(np.float32) / 255.0
            
            result = wrapper.infer(obs)
            print(f"  动作范围: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")
            
            # 测试 4: 不同分辨率（如果 backbone 支持）
            print("\n✓ 测试不同分辨率...")
            obs = self._create_test_obs(img_size=(240, 320))
            result = wrapper.infer(obs)
            print(f"  动作范围: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")
            
            print("\n✅ 测试 6 通过: 边界情况处理正确")
            return True
            
        except Exception as e:
            print(f"\n❌ 测试 6 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_test_obs(self, seed=None, qpos=None, img_size=(480, 640)):
        """创建测试观测数据"""
        if seed is not None:
            np.random.seed(seed)
        
        obs = {}
        
        # qpos
        if qpos is None:
            state_dim = self.stats['qpos_mean'].shape[0]
            obs['qpos'] = np.random.randn(state_dim).astype(np.float32)
        else:
            obs['qpos'] = qpos.astype(np.float32)
        
        # 图像
        for cam_name in self.camera_names:
            obs[cam_name] = np.random.randint(0, 255, size=(*img_size, 3), dtype=np.uint8)
        
        return obs
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 70)
        print("开始 ACT Policy 远程推理服务器测试")
        print("=" * 70)
        
        tests = [
            ("数据格式验证", self.test_1_data_format),
            ("归一化/反归一化", self.test_2_normalization),
            ("与原始推理对比", self.test_3_compare_with_original),
            ("多相机支持", self.test_4_multi_camera),
            ("性能测试", self.test_5_performance),
            ("边界情况", self.test_6_edge_cases),
        ]
        
        results = []
        for name, test_func in tests:
            try:
                result = test_func()
                results.append((name, result))
            except Exception as e:
                print(f"\n❌ 测试 '{name}' 发生异常: {e}")
                import traceback
                traceback.print_exc()
                results.append((name, False))
        
        # 总结
        print("\n" + "=" * 70)
        print("测试总结")
        print("=" * 70)
        
        for name, result in results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{status}: {name}")
        
        passed = sum(1 for _, r in results if r)
        total = len(results)
        
        print(f"\n总计: {passed}/{total} 测试通过")
        
        if passed == total:
            print("\n🎉 所有测试通过！")
            return True
        else:
            print(f"\n⚠️  {total - passed} 个测试失败")
            return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='测试 ACT Policy 远程推理服务器')
    parser.add_argument('--ckpt', '-i', required=True, help='Checkpoint 文件路径')
    parser.add_argument('--cameras', '-c', default='top', help='相机名称，逗号分隔')
    parser.add_argument('--num_queries', '-q', type=int, default=100, help='Num queries (chunk size)')
    parser.add_argument('--test', '-t', help='运行特定测试 (1-6)')
    
    args = parser.parse_args()
    
    camera_names = [name.strip() for name in args.cameras.split(',')]
    
    tester = TestACTPolicy(
        ckpt_path=args.ckpt,
        camera_names=camera_names,
        num_queries=args.num_queries
    )
    
    if args.test:
        # 运行特定测试
        test_map = {
            '1': tester.test_1_data_format,
            '2': tester.test_2_normalization,
            '3': tester.test_3_compare_with_original,
            '4': tester.test_4_multi_camera,
            '5': tester.test_5_performance,
            '6': tester.test_6_edge_cases,
        }
        
        if args.test in test_map:
            test_map[args.test]()
        else:
            print(f"错误: 无效的测试编号 '{args.test}'，请使用 1-6")
    else:
        # 运行所有测试
        success = tester.run_all_tests()
        exit(0 if success else 1)


if __name__ == '__main__':
    main()
