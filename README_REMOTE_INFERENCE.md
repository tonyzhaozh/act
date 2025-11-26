# ACT Policy 远程推理使用指南

本目录包含 ACT Policy 的远程推理服务器和客户端实现，支持通过 WebSocket 进行实时推理。

## 📁 文件说明

- `serve_act_policy.py` - ACT Policy 推理服务器
- `test_remote_inference.py` - 简单的推理测试客户端
- `act_remote_inference_with_temporal_agg.py` - 支持 Temporal Aggregation 的完整客户端示例

## 🚀 快速开始

### 1. 启动服务器

```bash
# 基本用法
python serve_act_policy.py -i <checkpoint_path> -p 8000

# 完整示例
python serve_act_policy.py \
    -i /path/to/policy_best.ckpt \
    -p 8000 \
    -c top,front \
    -q 100 \
    -d cuda
```

**参数说明：**
- `-i, --input`: checkpoint 文件路径（必需）
- `-p, --port`: 服务器端口（默认 8000）
- `-h, --host`: 服务器地址（默认 0.0.0.0）
- `-c, --camera_names`: 相机名称列表，逗号分隔（默认 'top'）
- `-q, --num_queries`: chunk size，即每次推理返回的动作序列长度（默认 100）
- `-d, --device`: 设备 'cuda' 或 'cpu'（默认 cuda）

**注意：** checkpoint 所在目录必须包含 `dataset_stats.pkl` 文件！

### 2. 运行客户端测试

```bash
# 简单测试
python test_remote_inference.py

# Temporal Aggregation 示例
python act_remote_inference_with_temporal_agg.py
```

## 📊 数据格式说明

### 输入格式（客户端 → 服务器）

```python
obs = {
    # 关节位置（必需）
    'qpos': np.ndarray,  # shape: (state_dim,), dtype: np.float32
                         # 例如: (14,) 表示 14 个关节的位置
    
    # 图像数据（根据相机配置）
    'top': np.ndarray,   # shape: (H, W, 3), dtype: np.uint8 或 np.float32
                         # 通道顺序: RGB（不是 BGR！）
                         # 例如: (480, 640, 3)
    
    'front': np.ndarray, # 可选，其他相机
}
```

### 输出格式（服务器 → 客户端）

```python
result = {
    'actions': np.ndarray,  # shape: (num_queries, action_dim)
                            # 例如: (100, 14) 表示 100 步动作序列
}
```

## 🔄 归一化处理说明

ACT Policy 的归一化在服务器端自动处理，无需客户端手动操作：

### 1. **qpos 归一化**（服务器端）
```python
# 输入: 原始关节位置
qpos_normalized = (qpos - qpos_mean) / qpos_std
```

### 2. **图像归一化**（服务器端）
```python
# 步骤1: 转换为 [0, 1]（如果是 uint8）
image = image.astype(np.float32) / 255.0

# 步骤2: ImageNet 标准归一化
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
image = normalize(image)
```

### 3. **动作反归一化**（服务器端）
```python
# 输出: 反归一化后的动作
action = action_normalized * action_std + action_mean
```

**客户端只需发送原始数据，归一化由服务器自动处理！**

## 📖 使用示例

### 基本推理

```python
from web_policy import WebSocketClientPolicy
import numpy as np

# 连接服务器
client = WebSocketClientPolicy(host='localhost', port=8000)

# 创建观测
obs = {
    'qpos': np.random.randn(14).astype(np.float32),
    'top': np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
}

# 推理
result = client.infer(obs)
actions = result['actions']  # (100, 14)

# 执行动作（逐步执行或按 query_frequency）
for i in range(len(actions)):
    action = actions[i]
    # 发送到机器人...
```

### Temporal Aggregation

```python
from act_remote_inference_with_temporal_agg import ACTRemoteInferenceClient

# 创建客户端（启用 temporal aggregation）
client = ACTRemoteInferenceClient(
    host='localhost',
    port=8000,
    temporal_agg=True,
    k=0.01  # 指数衰减系数
)

# Episode 循环
client.reset()
for t in range(max_timesteps):
    obs = get_observation()  # 获取当前观测
    action = client.get_action(obs, t)  # 获取融合后的动作
    execute_action(action)  # 执行动作
```

## ⚙️ Temporal Aggregation 原理

Temporal Aggregation 是 ACT 论文中的重要技术，用于提高动作的平滑性：

1. **每次推理返回 chunk**：模型预测未来 `num_queries` 步的动作序列
2. **收集历史预测**：对于第 t 步，收集所有曾经预测过该步的动作
3. **指数加权平均**：使用指数权重融合，越近期的预测权重越大

```python
# 权重计算
weights[i] = exp(-k * i)  # i 是距离当前的时间差
weights = weights / weights.sum()

# 加权平均
action_t = sum(predicted_actions[i] * weights[i])
```

**优势：**
- ✅ 动作更平滑，减少抖动
- ✅ 利用历史信息，提高鲁棒性
- ✅ 可以每步都查询，提高响应速度

## 🔧 故障排除

### 1. 找不到 dataset_stats.pkl

**错误：** `FileNotFoundError: 找不到 dataset_stats.pkl`

**解决：** 
- 确保 checkpoint 目录中有 `dataset_stats.pkl` 文件
- 该文件在训练时自动生成，包含归一化参数

### 2. 相机名称不匹配

**错误：** `ValueError: 缺少相机 'front' 的图像数据`

**解决：**
- 启动服务器时指定正确的相机名称：`-c top,front`
- 确保客户端发送的 obs 包含所有相机的图像

### 3. 图像通道顺序错误

**问题：** 推理结果不正常，颜色看起来不对

**解决：**
- 确保图像是 RGB 格式，不是 BGR
- 如果使用 OpenCV：`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`

### 4. shape 不匹配

**错误：** `RuntimeError: shape mismatch`

**解决：**
- 检查 qpos 维度是否与训练时一致
- 检查图像分辨率（通常是 480x640）
- 检查相机数量是否正确

## 📚 相关文档

- [Diffusion Policy 远程推理](../diffusion_policy/README_REMOTE_INFERENCE.md)
- [web_policy 库文档](../../web_policy/README.md)
- [ACT 论文](https://arxiv.org/abs/2304.13705)

## 💡 最佳实践

1. **使用 Temporal Aggregation**：在实际部署中强烈推荐，可以显著提高性能
2. **调整 query_frequency**：根据任务复杂度和计算资源调整
3. **监控推理延迟**：确保满足实时控制要求（通常 < 100ms）
4. **图像预处理**：在客户端做最小化预处理，减少网络传输
5. **批量推理**：如果控制多个机器人，可以使用批量推理提高效率

## 🎯 性能优化

- **GPU 推理**：使用 CUDA 加速，推理时间 ~20-50ms
- **TensorRT**：进一步优化可使用 TensorRT（需要额外工作）
- **减少图像分辨率**：如果可以接受，降低分辨率可以加速
- **网络优化**：使用局域网减少网络延迟
