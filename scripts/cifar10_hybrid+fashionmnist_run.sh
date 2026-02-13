./.venv/bin/python main.py --strategy hybrid_kd --dataset CIFAR10 --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 300 --seed 42 --num_workers 4
./.venv/bin/python main.py --strategy hybrid_kd --dataset CIFAR10 --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 300 --seed 3407 --num_workers 4
./.venv/bin/python main.py --strategy hybrid_kd --dataset CIFAR10 --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 300 --seed 2026 --num_workers 4

./.venv/bin/python main.py --dataset FashionMNIST --seed 42 --rounds 100 --num_workers 4
./.venv/bin/python main.py --dataset FashionMNIST --seed 3407 --rounds 100 --num_workers 4
./.venv/bin/python main.py --dataset FashionMNIST --seed 2026 --rounds 100 --num_workers 4

# ./.venv/bin/python main.py --dataset FashionMNIST --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 42 --rounds 100 --num_workers 4
./.venv/bin/python main.py --dataset FashionMNIST --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 3407 --rounds 100 --num_workers 4
./.venv/bin/python main.py --dataset FashionMNIST --strategy logit_kd --kd_T 1.0 --kd_alpha 0.5 --seed 2026 --rounds 100 --num_workers 4

./.venv/bin/python main.py --dataset FashionMNIST --strategy feature_kd --feat_alpha 0.5 --seed 42 --rounds 100 --num_workers 4
./.venv/bin/python main.py --dataset FashionMNIST --strategy feature_kd --feat_alpha 0.5 --seed 3407 --rounds 100 --num_workers 4
./.venv/bin/python main.py --dataset FashionMNIST --strategy feature_kd --feat_alpha 0.5 --seed 2026 --rounds 100 --num_workers 4

./.venv/bin/python main.py --dataset FashionMNIST --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 100 --seed 42 --num_workers 4
./.venv/bin/python main.py --dataset FashionMNIST --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 100 --seed 3407 --num_workers 4
./.venv/bin/python main.py --dataset FashionMNIST --strategy hybrid_kd --hybrid_bata 0.8 --kd_T 0.8 --kd_alpha 0.8 --feat_alpha 0.5 --rounds 100 --seed 2026 --num_workers 4
