#!/bin/bash
# 遇到错误即刻停止运行
set -e

echo "====================================================="
echo "开始运行补充实验: CIFAR-10 & FashionMNIST (C=4, C=6)"
echo "====================================================="

######################################################################
# 第一部分：CIFAR-10 数据集 (10分类, 300轮)
######################################################################

echo ">>> 开始训练 CIFAR-10 (client_classes = 4) <<<"

# 1. Baseline (FedAvg)
./.venv/bin/python main.py --dataset CIFAR10 --seed 42 --rounds 150 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --seed 3407 --rounds 150 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --seed 2026 --rounds 150 --num_workers 4 --client_classes 4 --num_classes 10

# 2. Logit KD
./.venv/bin/python main.py --dataset CIFAR10 --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 42 --rounds 150 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 3407 --rounds 150 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 2026 --rounds 150 --num_workers 4 --client_classes 4 --num_classes 10

# 3. Feature KD
./.venv/bin/python main.py --dataset CIFAR10 --strategy feature_kd --feat_alpha 0.5 --seed 42 --rounds 150 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --strategy feature_kd --feat_alpha 0.5 --seed 3407 --rounds 150 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --strategy feature_kd --feat_alpha 0.5 --seed 2026 --rounds 150 --num_workers 4 --client_classes 4 --num_classes 10

# 4. Hybrid KD
./.venv/bin/python main.py --dataset CIFAR10 --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 150 --seed 42 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 150 --seed 3407 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 150 --seed 2026 --num_workers 4 --client_classes 4 --num_classes 10


echo ">>> 开始训练 CIFAR-10 (client_classes = 6) <<<"

# 1. Baseline (FedAvg)
./.venv/bin/python main.py --dataset CIFAR10 --seed 42 --rounds 150 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --seed 3407 --rounds 150 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --seed 2026 --rounds 150 --num_workers 4 --client_classes 6 --num_classes 10

# 2. Logit KD
./.venv/bin/python main.py --dataset CIFAR10 --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 42 --rounds 150 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 3407 --rounds 150 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 2026 --rounds 150 --num_workers 4 --client_classes 6 --num_classes 10

# 3. Feature KD
./.venv/bin/python main.py --dataset CIFAR10 --strategy feature_kd --feat_alpha 0.5 --seed 42 --rounds 150 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --strategy feature_kd --feat_alpha 0.5 --seed 3407 --rounds 150 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --strategy feature_kd --feat_alpha 0.5 --seed 2026 --rounds 150 --num_workers 4 --client_classes 6 --num_classes 10

# 4. Hybrid KD
./.venv/bin/python main.py --dataset CIFAR10 --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 150 --seed 42 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 150 --seed 3407 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 150 --seed 2026 --num_workers 4 --client_classes 6 --num_classes 10


echo ">>> 开始训练 CIFAR-10 IID <<<"

# 1. Baseline (FedAvg)
./.venv/bin/python main.py --dataset CIFAR10 --partition_mode iid --seed 42 --rounds 150 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --partition_mode iid --seed 3407 --rounds 150 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --partition_mode iid --seed 2026 --rounds 150 --num_workers 4 --num_classes 10

# 2. Logit KD
./.venv/bin/python main.py --dataset CIFAR10 --partition_mode iid --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 42 --rounds 150 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --partition_mode iid --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 3407 --rounds 150 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --partition_mode iid --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 2026 --rounds 150 --num_workers 4 --num_classes 10

# 3. Feature KD
./.venv/bin/python main.py --dataset CIFAR10 --partition_mode iid --strategy feature_kd --feat_alpha 0.5 --seed 42 --rounds 150 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --partition_mode iid --strategy feature_kd --feat_alpha 0.5 --seed 3407 --rounds 150 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --partition_mode iid --strategy feature_kd --feat_alpha 0.5 --seed 2026 --rounds 150 --num_workers 4 --num_classes 10

# 4. Hybrid KD
./.venv/bin/python main.py --dataset CIFAR10 --partition_mode iid --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 150 --seed 42 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --partition_mode iid --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 150 --seed 3407 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset CIFAR10 --partition_mode iid --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 150 --seed 2026 --num_workers 4 --num_classes 10


######################################################################
# 第二部分：FashionMNIST 数据集 (10分类, 100轮)
######################################################################

echo ">>> 开始训练 FashionMNIST (client_classes = 4) <<<"

# 1. Baseline (FedAvg)
./.venv/bin/python main.py --dataset FashionMNIST --seed 42 --rounds 80 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --seed 3407 --rounds 80 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --seed 2026 --rounds 80 --num_workers 4 --client_classes 4 --num_classes 10

# 2. Logit KD
./.venv/bin/python main.py --dataset FashionMNIST --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 42 --rounds 80 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 3407 --rounds 80 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 2026 --rounds 80 --num_workers 4 --client_classes 4 --num_classes 10

# 3. Feature KD
./.venv/bin/python main.py --dataset FashionMNIST --strategy feature_kd --feat_alpha 0.5 --seed 42 --rounds 80 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --strategy feature_kd --feat_alpha 0.5 --seed 3407 --rounds 80 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --strategy feature_kd --feat_alpha 0.5 --seed 2026 --rounds 80 --num_workers 4 --client_classes 4 --num_classes 10

# 4. Hybrid KD
./.venv/bin/python main.py --dataset FashionMNIST --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 80 --seed 42 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 80 --seed 3407 --num_workers 4 --client_classes 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 80 --seed 2026 --num_workers 4 --client_classes 4 --num_classes 10


echo ">>> 开始训练 FashionMNIST (client_classes = 6) <<<"

# 1. Baseline (FedAvg)
./.venv/bin/python main.py --dataset FashionMNIST --seed 42 --rounds 80 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --seed 3407 --rounds 80 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --seed 2026 --rounds 80 --num_workers 4 --client_classes 6 --num_classes 10

# 2. Logit KD
./.venv/bin/python main.py --dataset FashionMNIST --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 42 --rounds 80 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 3407 --rounds 80 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 2026 --rounds 80 --num_workers 4 --client_classes 6 --num_classes 10

# 3. Feature KD
./.venv/bin/python main.py --dataset FashionMNIST --strategy feature_kd --feat_alpha 0.5 --seed 42 --rounds 80 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --strategy feature_kd --feat_alpha 0.5 --seed 3407 --rounds 80 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --strategy feature_kd --feat_alpha 0.5 --seed 2026 --rounds 80 --num_workers 4 --client_classes 6 --num_classes 10

# 4. Hybrid KD
./.venv/bin/python main.py --dataset FashionMNIST --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 80 --seed 42 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 80 --seed 3407 --num_workers 4 --client_classes 6 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 80 --seed 2026 --num_workers 4 --client_classes 6 --num_classes 10


echo ">>> 开始训练 FashionMNIST IID <<<"

# 1. Baseline (FedAvg)
./.venv/bin/python main.py --dataset FashionMNIST --partition_mode iid --seed 42 --rounds 80 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --partition_mode iid --seed 3407 --rounds 80 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --partition_mode iid --seed 2026 --rounds 80 --num_workers 4 --num_classes 10

# 2. Logit KD
./.venv/bin/python main.py --dataset FashionMNIST --partition_mode iid --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 42 --rounds 80 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --partition_mode iid --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 3407 --rounds 80 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --partition_mode iid --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 2026 --rounds 80 --num_workers 4 --num_classes 10

# 3. Feature KD
./.venv/bin/python main.py --dataset FashionMNIST --partition_mode iid --strategy feature_kd --feat_alpha 0.5 --seed 42 --rounds 80 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --partition_mode iid --strategy feature_kd --feat_alpha 0.5 --seed 3407 --rounds 80 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --partition_mode iid --strategy feature_kd --feat_alpha 0.5 --seed 2026 --rounds 80 --num_workers 4 --num_classes 10

# 4. Hybrid KD
./.venv/bin/python main.py --dataset FashionMNIST --partition_mode iid --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 80 --seed 42 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --partition_mode iid --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 80 --seed 3407 --num_workers 4 --num_classes 10
./.venv/bin/python main.py --dataset FashionMNIST --partition_mode iid --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 80 --seed 2026 --num_workers 4 --num_classes 10

echo "====================================================="
echo "所有补充实验运行完毕！"
echo "====================================================="
