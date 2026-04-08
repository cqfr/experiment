# Experiment: DP Federated Learning Baselines + Ours

本仓库实现统一可配置的联邦学习实验框架，支持：

- `FedAvg`
- `DP-FedAvg`
- `DP-FedSAM`
- `Ours`（Contrastive + Residual + Top-k + Adaptive Clipping + Heterogeneous Noise）

## 核心实现更新（重要）

当前 DP 加噪逻辑采用“先定聚合目标，再反推客户端噪声”：

1. 会计器先解每轮 `noise multiplier`：`z_t`
2. 计算聚合机制敏感度（加权聚合上界）：
   - `Δ2_t = p_max * C_t`
3. 得到聚合后目标噪声：
   - `sigma_agg = z_t * Δ2_t`
4. 反推各客户端基准噪声 `sigma_i`，满足：
   - `sum(p_i^2 * sigma_i^2) = sigma_agg^2`

默认策略为 `uniform`（最稳定），也支持 `heterogeneous`（按客户端重要性异构分配并做方差守恒重标定）。

另外，客户端参数级异构噪声已改为“平方均值归一化”，确保客户端内部噪声预算守恒。

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

# DP-FedAvg（客户端噪声均匀分配，推荐先跑这个）
python train.py --experiment dp_fedavg --dataset cifar10 --epsilon 8 --num_rounds 100 --client_noise_allocation uniform

# DP-FedSAM
python train.py --experiment dp_fedsam --dataset cifar10 --epsilon 8 --num_rounds 100 --client_noise_allocation uniform

# Ours
python train.py --experiment proposed --dataset cifar10 --epsilon 8 --topk_ratio 0.1 --num_rounds 100 --client_noise_allocation uniform

# Ours + importance matrix visualization
python train.py --experiment proposed --dataset cifar10 --epsilon 8 --topk_ratio 0.1 --num_rounds 100 --importance_viz --importance_viz_interval 5 --importance_viz_client 0
```

## 4. 关键参数

- `--experiment`: `fedavg | dp_fedavg | dp_fedsam | proposed`
- `--dataset`: `cifar10 | mnist`
- `--num_rounds`
- `--num_clients`
- `--clients_per_round`
- `--epsilon`
- `--topk_ratio`
- `--iid` / `--alpha`

DP 与噪声分配相关：

- `--client_noise_allocation`: `uniform | heterogeneous`
- `--client_variance_max_scale`: 异构客户端噪声上限倍数（默认 `10.0`）
- `--account_for_topk_in_q`: 开启后会计器 `q` 使用 `sampling_rate * topk_ratio`；默认关闭（仅用客户端采样率）

可视化相关：

- `--importance_viz`
- `--importance_viz_interval`
- `--importance_viz_client`
- `--importance_viz_max_elements`

## 5. 消融实验

```bash
# 全部实验
python run_ablation.py --dataset cifar10 --epsilon 8 --num_rounds 100

# 快速模式
python run_ablation.py --quick_test --dataset mnist

# 指定实验
python run_ablation.py --experiments proposed_full ablation_no_hetero_noise dp_fedsam
```

## 6. 输出目录

- `checkpoints/<exp>_<timestamp>/`
- `logs/<exp>_<timestamp>/`
- `results/<exp>_<timestamp>/`

日志包含：

- `history.json`
- `experiment_record.json`
- `config_summary.txt`

训练日志会打印：

- `eps`（累计隐私消耗）
- `z`（noise multiplier）
- `sigma_agg`（聚合后目标噪声标准差）
- `sigma_client`（本轮客户端基准噪声均值）

## 7. 最终提示词

可复用提示词见：`FINAL_PROMPT.md`。
