# CES-FL2: Differentially Private Federated Learning with Compression and Adaptive Clipping

这是一个基于 PyTorch 的联邦学习实验项目，面向 CIFAR-10 场景，重点研究以下三类机制的组合效果：

- 梯度压缩（Top-k）
- 自适应裁剪（Adaptive Clipping）
- 差分隐私噪声（同构/异构）

项目提供了完整训练入口、消融实验、超参数敏感性实验、快速模块自检和可视化脚本。

## 1. 项目目标

在非 IID 客户端数据分布下，对比并分析：

- `FedAvg`（无 DP）
- `DP-FedAvg`（标准 DP 基线）
- 本项目方法（压缩 + 自适应裁剪 + 异构噪声 + 对比正则）

核心关注指标：

- 测试精度（accuracy）
- 隐私预算消耗（epsilon）
- 裁剪阈值变化（clip history）

## 2. 代码结构

```text
ces-FL2/
├── train.py                  # 主训练流程（单次实验）
├── config.py                 # 所有配置与默认实验模板
├── quick_test.py             # 快速模块级自检
├── run_ablation.py           # 消融实验
├── run_hyperparam.py         # 超参数敏感性实验
├── visualize.py              # 结果可视化
├── client/
│   └── client.py             # 客户端训练/压缩/裁剪/加噪/上传
├── edge/
│   └── aggregator.py         # Edge 聚合与统计
├── server/
│   └── server.py             # 全局模型更新与裁剪阈值更新
├── compression/
│   └── topk.py               # 重要性评估 + Top-k 策略 + 残差累积
├── dp/
│   └── noise.py              # 隐私记账 + 噪声计算与注入
├── data/
│   └── utils.py              # CIFAR-10 加载 + IID/Dirichlet 划分
└── models/
    └── resnet.py             # GroupNorm 版 ResNet18
```

## 3. 环境准备

建议 Python 3.10+。

安装依赖：

```bash
pip install torch torchvision numpy matplotlib tqdm
```

说明：

- 数据集默认下载到 `data/datasets/`
- 首次运行会自动下载 CIFAR-10

## 4. 快速开始

### 4.1 先做模块自检

```bash
python quick_test.py
```

### 4.2 跑单次训练

```bash
# 无 DP 基线
python train.py --experiment fedavg --num_rounds 100

# DP-FedAvg 基线
python train.py --experiment dp_fedavg --epsilon 8.0 --num_rounds 100

# 本项目方法
python train.py --experiment proposed --epsilon 8.0 --topk_ratio 0.1 --num_rounds 100
```

常用参数：

- `--experiment`: `fedavg | dp_fedavg | proposed`
- `--num_rounds`: 通信轮数
- `--num_clients`: 客户端总数
- `--clients_per_round`: 每轮参与客户端数
- `--epsilon`: 总隐私预算
- `--topk_ratio`: Top-k 保留比例
- `--iid`: 使用 IID 划分（默认 Non-IID）
- `--alpha`: Dirichlet 参数（Non-IID 强度）

## 5. 复现实验

### 5.1 消融实验

```bash
# 全量消融
python run_ablation.py --epsilon 8.0 --num_rounds 100

# 快速测试模式（5轮）
python run_ablation.py --quick_test

# 仅运行指定实验
python run_ablation.py --experiments fedavg dp_fedavg proposed_full
```

### 5.2 超参数敏感性实验

```bash
# Top-k 比例敏感性
python run_hyperparam.py --experiment topk --num_rounds 100

# 隐私预算敏感性
python run_hyperparam.py --experiment epsilon --num_rounds 100

# Non-IID 强度敏感性
python run_hyperparam.py --experiment alpha --num_rounds 100

# 全部
python run_hyperparam.py --experiment all --num_rounds 100
```

## 6. 训练产物与日志

### 6.1 `train.py` 输出

每次运行会创建带时间戳目录：

- `./checkpoints/{experiment}_{timestamp}/`
- `./logs/{experiment}_{timestamp}/`

其中包括：

- `best_model.pt`
- `final_model.pt`
- `checkpoint_round_*.pt`（按 `save_interval`）
- `experiment_record.json`（完整记录）
- `history.json`（可视化兼容格式）
- `config_summary.txt`

### 6.2 `run_ablation.py` 与 `run_hyperparam.py` 输出

- 目录：`./results/...`
- 汇总：`ablation_summary_*.json` 或 `hyperparam/summary_*.json`

## 7. 可视化

### 7.1 批量可视化

```bash
python visualize.py --results_dir ./results --output_dir ./figures
```

### 7.2 单实验可视化

```bash
python visualize.py --history ./logs/<exp_timestamp>/history.json --plot_type all
```

`plot_type` 可选：

- `curves`
- `clip`
- `privacy`
- `all`

## 8. 核心机制说明

### 8.1 客户端（`client/client.py`）

每轮执行：

1. 接收全局模型
2. 本地训练（Standard / Contrastive / FedProx）
3. 残差累积
4. 参数重要性评估
5. Top-k 压缩
6. 梯度裁剪
7. 差分隐私加噪
8. 上传更新

### 8.2 压缩模块（`compression/topk.py`）

重要性策略：

- `fisher_grad`
- `grad_squared`
- `grad_normalized`

Top-k 策略：

- `global_topk`
- `layer_topk`
- `layer_norm_global`
- `weighted_layer_norm`（默认）

### 8.3 DP 模块（`dp/noise.py`）

- 隐私记账器：`PrivacyAccountant`
- 同构噪声：统一 `sigma`
- 异构噪声：按重要性分配 `epsilon_i`，逐参数计算 `sigma_i`

### 8.4 服务端与 Edge

- `edge/aggregator.py`：按数据量加权聚合与统计聚合
- `server/server.py`：更新全局模型，按策略更新裁剪阈值（`ADAPTIVE | EMA | QUANTILE`）

## 9. 常见问题

### Q1: 训练太慢

- 降低 `num_rounds`
- 降低 `client.local_epochs`
- 减少 `clients_per_round`

### Q2: 显存/内存不足

- 降低 `batch_size`
- 减少每轮参与客户端数

### Q3: 隐私预算消耗太快

- 增大 `epsilon_total`
- 减小 `topk_ratio`
- 保持 `use_subsampling_amplification` 与 `use_sparsity_amplification` 为 `True`

### Q4: Non-IID 下精度偏低

- 优先使用 `CONTRASTIVE`
- 调整 `alpha`（数据划分）与对比正则超参（`client.alpha`, `client.beta`）

## 10. 清理说明

本次已移除无用缓存目录（`__pycache__/`），并在 `.gitignore` 中加入以下忽略项，防止再次污染仓库：

- `__pycache__/`
- `*.py[cod]`
- `checkpoints/`
- `logs/`
- `results/`
- `figures/`
- `data/datasets/`
