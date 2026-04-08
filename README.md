# Experiment: DP Federated Learning Baselines + Ours

本仓库实现了统一可配置的联邦学习实验框架，支持以下方法：

- `FedAvg`
- `DP-FedAvg`
- `DP-FedSAM`
- `Ours`（Contrastive + Residual + Top-k + Adaptive Clipping + Heterogeneous Noise）

核心更新：

- 客户端上传统一为 `Delta_w = w_local - w_global`
- DP 采用 `RDP accountant` 逐轮反解 `sigma_base`
- 异构噪声改为“相对比例 + 统一缩放满足每轮预算”
- Contrastive 正则使用有界形式，避免发散
- 数据集支持 `CIFAR-10` 和 `MNIST`

## 1. 环境安装

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## 2. 快速检查

```bash
python quick_test.py
```

## 3. 单次训练

```bash
# FedAvg
python train.py --experiment fedavg --dataset cifar10 --num_rounds 100

# DP-FedAvg
python train.py --experiment dp_fedavg --dataset cifar10 --epsilon 8 --num_rounds 100

# DP-FedSAM
python train.py --experiment dp_fedsam --dataset cifar10 --epsilon 8 --num_rounds 100

# Ours
python train.py --experiment proposed --dataset cifar10 --epsilon 8 --topk_ratio 0.1 --num_rounds 100

# Ours + importance matrix visualization (every 5 rounds, client 0)
python train.py --experiment proposed --dataset cifar10 --epsilon 8 --topk_ratio 0.1 --num_rounds 100 --importance_viz --importance_viz_interval 5 --importance_viz_client 0
```

`train.py` 关键参数：

- `--experiment`: `fedavg | dp_fedavg | dp_fedsam | proposed`
- `--dataset`: `cifar10 | mnist`
- `--importance_viz`: save importance heatmaps to `logs/<exp>/importance_viz/`
- `--importance_viz_interval`: visualization interval by round
- `--importance_viz_client`: target client id for visualization
- `--num_rounds`
- `--num_clients`
- `--clients_per_round`
- `--epsilon`
- `--topk_ratio`
- `--iid` / `--alpha`

## 4. 消融实验

```bash
# 全部实验
python run_ablation.py --dataset cifar10 --epsilon 8 --num_rounds 100

# 快速模式
python run_ablation.py --quick_test --dataset mnist

# 只跑指定实验
python run_ablation.py --experiments proposed_full ablation_no_hetero_noise dp_fedsam
```

## 5. 输出目录

- `checkpoints/<exp>_<timestamp>/`
- `logs/<exp>_<timestamp>/`
- `results/<exp>_<timestamp>/`

日志包含：

- `history.json`
- `experiment_record.json`
- `config_summary.txt`

## 6. 最终提示词

最终可复用提示词见：`FINAL_PROMPT.md`。
