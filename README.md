# Fed

一个用于联邦学习与知识蒸馏实验的项目。核心训练逻辑主要位于 `src/`，根目录下的 `1-download_dataset.py`、`2-train_teacher_cnn.py`、`6-split_dataset.py`、`main.py` 对应完整实验流程，`scripts/` 里的脚本主要是部分批量运行示例，不是全部核心代码。

## 项目结构

```text
Fed/
├── 1-download_dataset.py      # 下载数据集
├── 2-train_teacher_cnn.py     # 预训练教师模型
├── 6-split_dataset.py         # 生成联邦划分文件
├── main.py                    # 正式联邦训练入口
├── src/
│   ├── configs/               # 默认配置
│   ├── data/                  # 数据集与划分逻辑
│   ├── models/                # Teacher / Student / Adapter 模型
│   ├── server/                # 联邦训练服务端与 checkpoint
│   ├── trainers/              # FedAvg / KD 等训练策略
│   └── utils/                 # 日志等工具
└── scripts/                   # 部分批量运行脚本示例
```

## 环境准备

项目使用 Python 运行，依赖定义在 `pyproject.toml` 中，主要包括：`torch`、`torchvision`、`numpy`、`pandas`、`matplotlib`、`tensorboard`。

如果你使用 `uv`：

```bash
uv sync
```

如果你使用已有虚拟环境，也可以按 `pyproject.toml` 中的依赖自行安装。

## 完整流程

推荐按照下面顺序执行：

1. 下载数据集
2. 预训练教师模型
3. 生成联邦数据划分
4. 启动正式联邦训练

---

## 1. 下载数据集

下载脚本为 `1-download_dataset.py`：

```bash
python 1-download_dataset.py
```

默认会把数据下载到项目根目录下的 `data/` 中。

当前脚本显式下载的是：

1. `MNIST`
2. `CIFAR-10`

说明：

1. `CIFAR100`、`FashionMNIST` 虽然没有在这个脚本里显式列出，但项目在后续划分或训练时如果发现本地不存在，也会通过 `torchvision` 自动下载。
2. 下载完成后的原始数据通常位于 `data/` 目录下。

---

## 2. 预训练教师模型

教师模型预训练脚本为 `2-train_teacher_cnn.py`：

```bash
python 2-train_teacher_cnn.py
```

这个脚本会：

1. 加载指定数据集
2. 构建 `TeacherCNN`
3. 进行常规监督训练
4. 保存最优权重和阶段性结果

### 当前脚本默认行为

`2-train_teacher_cnn.py` 的 `main()` 当前默认执行的是：

```python
run_experiment("cifar100", "teacher_cnn", epochs=300, num_classes=100)
```

也就是说，直接运行时默认预训练的是 `CIFAR100` 教师模型。

如果你想训练其他数据集，需要修改 `main()` 中对应的 `run_experiment(...)` 调用，例如：

```python
run_experiment("cifar10", "teacher_cnn", epochs=300)
run_experiment("mnist", "teacher_cnn", epochs=30)
run_experiment("fashionmnist", "teacher_cnn", epochs=50)
```

### 预训练输出

训练完成后，结果通常会输出到：

1. `weights/`：教师模型权重
2. `results/`：训练过程指标 CSV

其中权重命名与正式训练加载逻辑相关，例如正式联邦训练会尝试加载：

```text
weights/{dataset小写}_teacher_cnn_best.pth
```

例如：

1. `weights/cifar10_teacher_cnn_best.pth`
2. `weights/cifar100_teacher_cnn_best.pth`

如果你运行的是蒸馏策略（`logit_kd`、`feature_kd`、`hybrid_kd`），建议先完成对应数据集的教师模型预训练，否则系统会退化为加载随机教师权重。

---

## 3. 生成联邦数据划分

数据划分脚本为 `6-split_dataset.py`：

```bash
python 6-split_dataset.py --dataset CIFAR100 --num_users 10 --partition_mode pathological --partition_seed 42
```

该脚本会在 `data/partitions/` 下生成划分文件，供正式联邦训练使用。

### 常用参数

1. `--dataset`：数据集名称，支持 `MNIST`、`CIFAR10`、`FashionMNIST`、`CIFAR100`
2. `--num_users`：客户端数量
3. `--partition_mode`：划分方式，当前支持 `iid` 和 `pathological`
4. `--partition_seed`：划分随机种子
5. `--client_classes`：在 `pathological` 模式下，每个客户端拥有的类别数
6. `--force`：若目标划分文件已存在，是否强制重建

### 划分方式说明

1. `iid`：尽量均匀随机分给各个客户端
2. `pathological`：按类别分片，制造更典型的非 IID 场景

如果 `pathological` 模式下没有手动指定 `--client_classes`，项目会按数据集使用默认值：

1. `MNIST` / `CIFAR10` / `FashionMNIST` 默认每个客户端 `2` 类
2. `CIFAR100` 默认每个客户端 `10` 类

示例：

```bash
python 6-split_dataset.py --dataset CIFAR10 --num_users 10 --partition_mode pathological --partition_seed 42 --client_classes 2
```

生成的划分文件类似：

```text
data/partitions/pathological_CIFAR10_k10_c2_seed42.pkl
```

说明：正式训练阶段如果找不到对应划分文件，也会尝试自动生成；但为了保证实验可复现，仍然建议先显式执行一次划分脚本。

---

## 4. 正式联邦训练

正式训练入口为 `main.py`。

最简单的运行方式：

```bash
python main.py
```

默认配置位于 `src/configs/config.py`，当前默认策略是：

1. `strategy="fedavg"`
2. `dataset="CIFAR10"`
3. `rounds=100`
4. `num_users=10`

### 支持的训练策略

`main.py` 当前支持以下策略：

1. `fedavg`
2. `logit_kd`
3. `feature_kd`
4. `hybrid_kd`

### 常用命令示例

#### 4.1 FedAvg

```bash
python main.py --dataset CIFAR10 --strategy fedavg --rounds 100 --num_users 10 --seed 42
```

#### 4.2 Logit KD

```bash
python main.py --dataset CIFAR10 --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --rounds 100 --num_users 10 --seed 42
```

#### 4.3 Feature KD

```bash
python main.py --dataset CIFAR10 --strategy feature_kd --feat_alpha 1.0 --rounds 100 --num_users 10 --seed 42
```

#### 4.4 Hybrid KD

```bash
python main.py --dataset CIFAR10 --strategy hybrid_kd --hybrid_bata 0.5 --kd_T 1.0 --kd_alpha 0.5 --feat_alpha 1.0 --rounds 100 --num_users 10 --seed 42
```

### 常用训练参数

1. `--dataset`：数据集名称
2. `--strategy`：训练策略
3. `--seed`：随机种子
4. `--rounds`：联邦通信轮数
5. `--num_users`：客户端总数
6. `--frac`：每轮参与训练的客户端比例
7. `--local_ep`：客户端本地训练 epoch 数
8. `--local_bs`：客户端本地 batch size
9. `--lr`：学习率
10. `--momentum`：SGD 动量
11. `--partition_mode`：数据划分方式
12. `--partition_seed`：数据划分随机种子
13. `--partition_path`：手动指定划分文件路径

蒸馏策略额外参数：

1. `--kd_T`：logit 蒸馏温度
2. `--kd_alpha`：logit 蒸馏损失权重
3. `--feat_alpha`：特征蒸馏损失权重
4. `--hybrid_bata`：混合蒸馏权重
5. `--student_channels`：学生特征通道数
6. `--teacher_channels`：教师特征通道数

### 训练输出

正式训练会把结果主要写到：

1. `results/unified_logs/`：每轮训练日志 CSV
2. `results/unified_logs/checkpoints/`：中间检查点
3. `weights/`：教师模型权重目录

项目支持断点续训，`main.py` 中默认：

```bash
python main.py --resume auto
```

含义是自动查找同配置下最近的 checkpoint 并继续训练；如果不希望自动续训，可以显式指定：

```bash
python main.py --resume none
```

---

## 一个推荐实验流程

以 `CIFAR100 + hybrid_kd` 为例：

```bash
python 1-download_dataset.py
python 2-train_teacher_cnn.py
python 6-split_dataset.py --dataset CIFAR100 --num_users 10 --partition_mode pathological --partition_seed 42 --client_classes 10
python main.py --dataset CIFAR100 --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 300 --seed 42 --num_workers 4 --client_classes 10 --num_classes 100
```

如果你想批量运行多组实验，可以参考 `scripts/` 中已有脚本，例如 `scripts/cifar100_run.sh`。这些脚本主要是命令组合示例，真正的训练逻辑仍在 `src/` 和根目录入口脚本中。

---

## 代码说明

1. `src/models/`：定义教师模型、学生模型和特征适配器
2. `src/trainers/`：定义各类本地训练器，如 `FedAvgTrainer`、KD 训练器等
3. `src/server/worker.py`：联邦训练主流程、客户端并行训练、模型聚合与日志记录
4. `src/data/partition.py`：数据划分生成与读取
5. `src/configs/config.py`：默认实验配置

如果你是第一次阅读这个项目，建议优先按以下顺序看代码：

1. `main.py`
2. `src/configs/config.py`
3. `src/server/worker.py`
4. `src/trainers/`
5. `src/data/partition.py`

## 补充说明

1. 数据集名称在不同脚本中大小写风格略有不同，正式联邦训练建议使用 `CIFAR10`、`MNIST`、`FashionMNIST`、`CIFAR100` 这几种写法。
2. 蒸馏实验依赖教师模型权重文件名与数据集匹配，开始正式训练前请确认 `weights/` 下已存在对应的 `*_teacher_cnn_best.pth`。
3. `scripts/` 目录不是项目全部逻辑，只是部分实验命令的快捷入口。
4. 本实验仅在单卡 3060 Laptop 环境下测试，为在多卡环境下测试。
