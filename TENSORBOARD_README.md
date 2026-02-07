# TensorBoard 集成说明

## 概述

已将 `CachedKDTrainer` 和 `BaseTrainer` 修改为支持 TensorBoard 监控，`compute_loss` 方法现在返回包含详细损失组件的字典。

## 主要修改

### 1. BaseTrainer 修改

- **TensorBoard 集成**: 在 `__init__` 中初始化 `SummaryWriter`
- **compute_loss 返回类型**: 从 `tuple[Any, Any]` 改为 `dict[str, Any]`
- **自动记录**: 在训练循环中自动记录所有损失组件到 TensorBoard

### 2. CachedKDTrainer 修改

- **compute_loss 返回字典**: 包含以下键
  - `loss`: 总损失（必需）
  - `outputs`: 模型输出（必需）
  - `ce_loss`: 交叉熵损失
  - `kd_loss`: 知识蒸馏损失（logit_kd 和 hybrid_kd）
  - `feat_loss`: 特征损失（feature_kd）
  - `fd_loss`: 特征蒸馏损失（hybrid_kd）
  - `distillation_part`: 蒸馏部分总损失（hybrid_kd）

### 3. 辅助方法修改

- `_loss_kd_logit`: 返回 `(total_loss, ce_loss, kd_loss)`
- `_loss_kd_feature`: 返回 `(total_loss, ce_loss, feat_loss)`

## TensorBoard 监控指标

### Batch 级别指标
- `Loss/batch`: 每个 batch 的总损失
- `Loss/ce_loss`: 交叉熵损失
- `Loss/kd_loss`: KD 损失（如果适用）
- `Loss/feat_loss`: 特征损失（如果适用）
- `Loss/fd_loss`: 特征蒸馏损失（如果适用）
- `Loss/distillation_part`: 蒸馏部分（如果适用）

### Epoch 级别指标
- `Loss/epoch`: 每个 epoch 的平均损失
- `Accuracy/epoch`: 每个 epoch 的准确率
- `Learning_Rate`: 学习率变化

## 使用方法

### 1. 配置日志目录

在配置中添加 `log_dir` 参数（可选，默认为 "runs"）：

```python
config = {
    "log_dir": "runs",  # TensorBoard 日志目录
    # ... 其他配置
}
```

### 2. 训练模型

正常训练模型，TensorBoard 日志会自动记录到 `{log_dir}/client_{client_id}` 目录。

### 3. 查看 TensorBoard

使用提供的脚本启动 TensorBoard：

```bash
./view_tensorboard.sh [log_dir]
```

或者手动启动：

```bash
tensorboard --logdir=runs --port=6006
```

然后在浏览器中访问 `http://localhost:6006`

## 日志目录结构

```
runs/
├── client_0/
│   └── events.out.tfevents.*
├── client_1/
│   └── events.out.tfevents.*
└── ...
```

每个客户端都有独立的日志目录，便于对比不同客户端的训练过程。

## 注意事项

1. **其他 Trainer 需要更新**: 如果项目中还有其他继承自 `BaseTrainer` 的类（如 `FeatureKDTrainer`、`FedKDHybridTrainer`），它们的 `compute_loss` 方法也需要修改为返回字典格式。

2. **磁盘空间**: TensorBoard 日志会占用一定的磁盘空间，建议定期清理旧的日志文件。

3. **性能影响**: 记录到 TensorBoard 会有轻微的性能开销，但通常可以忽略不计。

## 示例：查看训练曲线

启动 TensorBoard 后，你可以：

1. **对比不同策略**: 在 SCALARS 标签页中选择多个客户端进行对比
2. **监控损失组件**: 查看 CE loss、KD loss、Feature loss 等各个组件的变化
3. **分析学习率**: 查看学习率调度器的效果
4. **评估收敛性**: 通过损失和准确率曲线判断模型是否收敛

## 故障排除

### 问题：TensorBoard 无法启动

确保已安装 TensorBoard：
```bash
pip install tensorboard
```

### 问题：看不到日志

检查日志目录是否正确，以及训练是否已经开始。

### 问题：日志文件过大

可以使用以下命令清理旧日志：
```bash
rm -rf runs/*
```
