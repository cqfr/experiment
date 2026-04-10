# 联邦学习 + 差分隐私实验项目说明（详细版）

本仓库实现了同一套可配置框架下的 4 类实验方法，并支持完整训练、消融、超参数敏感性分析与可视化：

- `FedAvg`
- `DP-FedAvg`
- `DP-FedSAM`
- `Proposed`（对比正则 + 残差累积 + Top-k 压缩 + 自适应裁剪 + 异构噪声）

---

## 1. 项目文件遍历结果

### 1.1 源码与文档文件（手写）

- `.gitignore`
- `FINAL_PROMPT.md`
- `README.md`
- `__init__.py`
- `config.py`
- `train.py`
- `run_ablation.py`
- `run_hyperparam.py`
- `quick_test.py`
- `visualize.py`
- `client/__init__.py`
- `client/client.py`
- `server/__init__.py`
- `server/server.py`
- `edge/__init__.py`
- `edge/aggregator.py`
- `models/__init__.py`
- `models/resnet.py`
- `data/__init__.py`
- `data/utils.py`
- `compression/__init__.py`
- `compression/topk.py`
- `dp/__init__.py`
- `dp/noise.py`

### 1.2 数据文件（非代码）

- `data/datasets/cifar-10-python.tar.gz`
- `data/datasets/MNIST/raw/train-images-idx3-ubyte.gz`
- `data/datasets/cifar-10-batches-py/`（CIFAR-10 解压目录，当前环境存在访问限制）

### 1.3 自动生成文件（运行产物）

- 各目录下 `__pycache__/*.pyc`
- 训练时生成：`checkpoints/`、`logs/`、`results/`、`figures/`

### 1.4 版本管理内部文件

- `.git/` 下对象、日志、引用、hooks 为 Git 内部元数据，不包含实验业务逻辑。

---

## 2. 实验怎么做（复现步骤）

## 2.1 环境安装

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## 2.2 快速连通性测试

```bash
python quick_test.py
```

目标：验证配置、隐私会计、客户端上传与服务端聚合主链路可执行。

## 2.3 单次训练

```bash
# FedAvg
python train.py --experiment fedavg --dataset cifar10 --num_rounds 100

# DP-FedAvg（建议先跑这个）
python train.py --experiment dp_fedavg --dataset cifar10 --epsilon 8 --num_rounds 100 --client_noise_allocation uniform

# DP-FedSAM
python train.py --experiment dp_fedsam --dataset cifar10 --epsilon 8 --num_rounds 100 --client_noise_allocation uniform

# Proposed
python train.py --experiment proposed --dataset cifar10 --epsilon 8 --topk_ratio 0.1 --num_rounds 100 --client_noise_allocation uniform

# Proposed + 重要性可视化
python train.py --experiment proposed --dataset cifar10 --epsilon 8 --topk_ratio 0.1 --importance_viz --importance_viz_interval 5 --importance_viz_client 0
```

## 2.4 消融实验

```bash
# 全部消融 + 基线
python run_ablation.py --dataset cifar10 --epsilon 8 --num_rounds 100

# 快速测试
python run_ablation.py --quick_test --dataset mnist

# 指定实验子集
python run_ablation.py --experiments proposed_full ablation_no_hetero_noise dp_fedsam
```

## 2.5 超参数敏感性实验

```bash
# 全部敏感性实验（k, epsilon, alpha）
python run_hyperparam.py --experiment all --num_rounds 100

# 分开跑
python run_hyperparam.py --experiment topk
python run_hyperparam.py --experiment epsilon
python run_hyperparam.py --experiment alpha

# 快速模式
python run_hyperparam.py --experiment all --quick_test
```

## 2.6 可视化

```bash
# 批量扫描 results 目录生成图
python visualize.py --results_dir ./results --output_dir ./figures --plot_type all

# 画单个 history.json
python visualize.py --history ./logs/xxx/history.json --output_dir ./figures --plot_type curves
python visualize.py --history ./logs/xxx/history.json --output_dir ./figures --plot_type clip
python visualize.py --history ./logs/xxx/history.json --output_dir ./figures --plot_type privacy
```

---

## 3. 训练主流程（代码级）

单轮训练逻辑（`train.py -> FLTrainer.train_round`）：

1. 服务端给出当前 `clip_norm`。
2. 按客户端采样率 `q`（可选乘 `topk_ratio`）计算会计采样率。
3. 从 `num_clients` 随机采样 `clients_per_round` 个客户端。
4. 若启用 DP：
   - 用 `RDPAccountant.solve_noise_multiplier_for_round` 反解本轮 `z`。
   - 用 `compute_weighted_sensitivity` 得到加权聚合敏感度上界。
   - 构造目标聚合标准差 `sigma_agg = z_base * sensitivity`。
   - 用 `allocate_client_noise_stds` 反解每个客户端的 `sigma_base`。
5. 每个客户端执行 `train_and_upload`：本地训练 -> 残差累积 -> 重要性计算 -> Top-k -> 裁剪 -> 加噪。
6. 服务端 `aggregate` 做加权聚合，`update_global_model` 更新全局参数并更新裁剪阈值。
7. 评测测试集，记录 `accuracy/loss/clip/upload/epsilon/z/sigma` 等日志。
8. 按需保存 checkpoint、history、experiment_record。

---

## 4. 配置系统说明（`config.py`）

### 4.1 枚举类型

- `TrainingStrategy`: `STANDARD / CONTRASTIVE / FEDPROX / DP_FEDSAM`
- `ImportanceStrategy`: `FISHER_GRAD / GRAD_SQUARED / GRAD_NORMALIZED`
- `TopKStrategy`: `GLOBAL_TOPK / LAYER_TOPK / LAYER_NORM_GLOBAL / WEIGHTED_LAYER_NORM`
- `LayerWeightMethod`: `MEAN / MEDIAN / TOTAL_SUM / TRIMMED_MEAN`
- `StatType`: `BINARY / L2_NORM`
- `StatsAggMethod`: `QUANTILE / ALL`
- `ClipUpdateMethod`: `ADAPTIVE / EMA`
- `DownlinkStrategy`: `FULL / TOPK`

### 4.2 核心配置 dataclass

- `ClientConfig`: 本地训练、正则、SAM、重要性、Top-k、统计上传等配置。
- `EdgeConfig`: 边缘统计聚合方式。
- `ServerConfig`: 服务端学习率、裁剪阈值更新、下行策略。
- `DPConfig`: 隐私预算、RDP 阶数、异构噪声、客户端噪声分配策略。
- `ExperimentConfig`: 全局实验配置，组合了 client/edge/server/dp。

### 4.3 预设配置函数

- `get_fedavg_config()`: 纯 FedAvg，无 DP，无压缩残差。
- `get_dp_fedavg_config(epsilon)`: DP-FedAvg，同方差 DP 噪声。
- `get_dp_fedsam_config(epsilon)`: DP-FedSAM，在 DP-FedAvg 上替换本地优化为 SAM 两步。
- `get_proposed_config(epsilon)`: Proposed 全模块默认开启。

---

## 5. 文件与函数块详解

## 5.1 根目录文件

### `__init__.py`

作用：包级导出与版本元信息。

- 导出：配置枚举、dataclass 与预设函数（便于外部 `import`）。
- `__version__ = "1.1.0"`
- `__author__ = "HUST Thesis Project"`

### `.gitignore`

作用：忽略数据集、缓存和实验产物。

- `data/datasets/`
- `__pycache__/`
- `*.py[cod]`
- `checkpoints/ logs/ results/ figures/`

### `FINAL_PROMPT.md`

作用：实验需求与工程约束说明文档（相当于研究任务清单）。

### `README.md`

作用：项目总说明与开发/实验导航（当前文件）。

---

## 5.2 训练入口与实验脚本

### `train.py`

作用：主训练入口，负责端到端联邦训练编排。

类与函数块：

- `FLTrainer.__init__(config)`
  - 初始化随机种子、设备、数据加载器、模型、服务端、客户端列表、隐私会计器、历史记录。
- `FLTrainer._set_seed(seed)`
  - 固定 `random/numpy/torch` 随机种子。
- `FLTrainer._select_clients(count)`
  - 每轮随机采样客户端 ID。
- `FLTrainer.train_round(round_num)`
  - 单轮核心逻辑：噪声标定、客户端训练上传、服务端聚合更新、可视化快照、隐私预算累积。
- `FLTrainer._save_importance_visualization(...)`
  - 将某客户端的重要性/掩码向量保存为 `.npz` 与热图 `.png`。
- `FLTrainer.evaluate()`
  - 调用服务端在测试集上评测。
- `FLTrainer.train()`
  - 轮次循环，含日志打印、早停（隐私预算耗尽）、checkpoint 保存、history 落盘。
- `FLTrainer._save_checkpoint(round_num, is_best, is_final)`
  - 保存模型参数、轮次、裁剪阈值、配置快照。
- `FLTrainer._save_history()`
  - 写出 `history.json`、`experiment_record.json`、`config_summary.txt`。
- `FLTrainer._config_to_dict(config)`
  - dataclass + 枚举转可序列化字典。
- `run_experiment(experiment_name, config)`
  - 配置日志/模型目录并执行训练。
- `parse_args()`
  - 解析命令行参数。
- `main()`
  - 根据 `--experiment` 选择预设并覆盖参数后执行。

### `run_ablation.py`

作用：统一运行基线 + 消融组合。

函数块：

- `_base_overrides(config, num_rounds, dataset)`
  - 给任意配置打上统一实验底座（客户端数、采样数、alpha 等）。
- `get_ablation_set(epsilon, num_rounds, dataset)`
  - 构造实验集合：
  - 基线：`fedavg / dp_fedavg / dp_fedsam / proposed_full`
  - 消融：`ablation_no_compression / ablation_no_adaptive_clip / ablation_no_hetero_noise / ablation_no_contrastive`
- `run_one(name, config)`
  - 跑单个实验并返回摘要。
- `run_ablation_study(...)`
  - 批量执行并保存总表 `ablation_summary_*.json`。
- `parse_args()`
  - 命令行参数解析。
- `main()`
  - 入口函数（支持 `--quick_test`）。

### `run_hyperparam.py`

作用：超参数敏感性分析脚本。

函数块：

- `get_base_proposed_config()`
  - 返回 Proposed 的基础可复用配置模板。
- `run_topk_sensitivity(k_values, num_rounds)`
  - 扫描 `topk_ratio` 对精度影响。
- `run_epsilon_sensitivity(epsilon_values, num_rounds)`
  - 扫描隐私预算 `epsilon_total`。
- `run_alpha_sensitivity(alpha_values, num_rounds)`
  - 扫描 Dirichlet `alpha`（数据异构程度）。
- `run_all_hyperparam_experiments(...)`
  - 串行执行全部敏感性实验并保存汇总 JSON。
- 文件底部 `if __name__ == "__main__":`
  - 解析 CLI 参数并分支执行。

### `quick_test.py`

作用：快速烟雾测试（不做完整实验）。

函数块：

- `test_config()`
  - 校验不同预设配置是否正确生成。
- `test_dp_accountant()`
  - 校验 RDP 会计器能求出噪声并消费一轮。
- `test_client_server_round()`
  - 用 `TinyNet` 模拟一轮“客户端上传 + 服务端聚合更新”。
- `run_all()`
  - 依次跑上述测试并输出 `quick_test passed`。

### `visualize.py`

作用：绘图工具，面向 history/result JSON。

函数块：

- `plot_training_curves(history_paths, save_path, metric)`
  - 多实验精度/损失曲线对比。
- `plot_privacy_accuracy_tradeoff(results, save_path)`
  - 隐私预算-精度权衡图。
- `plot_ablation_bar(results, save_path)`
  - 消融柱状图。
- `plot_sensitivity_curve(results, param_name, param_key, save_path)`
  - 敏感性折线图。
- `plot_clip_history(history_path, save_path)`
  - 裁剪阈值随轮次变化图。
- `plot_privacy_consumption(history_path, epsilon_total, save_path)`
  - 隐私预算消耗曲线。
- `generate_report(results_dir, output_dir)`
  - 扫描结果目录并生成批量图。
- 文件底部 `if __name__ == "__main__":`
  - CLI 入口，支持单实验/批量可视化。

---

## 5.3 客户端模块

### `client/__init__.py`

作用：导出 `FLClient`、`ClientUpdate`。

### `client/client.py`

作用：客户端本地训练、压缩、裁剪、DP 加噪、上传打包。

类与函数块：

- `ClientUpdate`（dataclass）
  - 客户端上传载荷：`delta_w/data_size/stat/clipped/noise_sigma/upload_ratio/...`。
- `FLClient.__init__(...)`
  - 深拷贝模型、绑定数据、初始化残差器、Fisher 缓存、随机数生成器。
- `FLClient.receive_global_model(global_weights)`
  - 接收并覆盖本地模型参数。
- `FLClient.local_train()`
  - 本地训练并返回 `Delta_w = w_local - w_global`。
- `FLClient._train_standard(optimizer, criterion)`
  - 标准 SGD 训练；可叠加 Contrastive/FedProx 正则。
- `FLClient._train_dp_fedsam(optimizer, criterion)`
  - SAM 两步更新：先扰动再反向。
- `FLClient._grad_l2_norm()`
  - 计算当前梯度整体 L2 范数。
- `FLClient._regularization_loss()`
  - 根据策略计算正则项：
  - FedProx：`mu * ||w - w_global||^2`
  - Contrastive：`alpha*||w-wg||^2 + beta*max(0,m-||w-w_old||)^2`
- `FLClient.accumulate_residual(delta_w)`
  - Error-feedback 残差累积。
- `FLClient.compute_importance(delta_w)`
  - 按配置计算重要性（Fisher*|g| / |g|^2 / 层内归一化）。
- `FLClient.topk_compress(delta_w, importance)`
  - 调用统一压缩接口并统计上传比例。
- `FLClient.clip_delta(delta_w, clip_norm)`
  - L2 裁剪上传更新。
- `FLClient.add_dp_noise(delta_w, masks, importance, sigma_base)`
  - DP 噪声入口：同方差或异构噪声。
- `FLClient._add_heterogeneous_noise(...)`
  - 将分层参数展平后按重要性分配相对噪声并注入。
- `FLClient.train_and_upload(...)`
  - 客户端完整流水线入口，输出 `ClientUpdate`。
- `FLClient._build_vector_snapshot(tensors, max_elements)`
  - 将字典参数展平并限长采样，用于重要性可视化。

---

## 5.4 服务端与边缘聚合模块

### `server/__init__.py`

作用：导出 `FLServer`。

### `server/server.py`

作用：服务端聚合、全局模型更新、裁剪阈值更新、评估。

类与函数块：

- `AggregatedResult`（dataclass）
  - 聚合输出载荷（全局更新 + 聚合统计）。
- `FLServer.__init__(model, config, device)`
  - 初始化全局模型、裁剪阈值、历史。
- `FLServer.get_global_weights()`
  - 返回全局模型参数快照。
- `FLServer.get_clip_norm()`
  - 返回当前裁剪阈值。
- `FLServer.aggregate(client_updates, data_sizes, stats, stats_agg_method)`
  - 封装“参数加权聚合 + 统计聚合”。
- `FLServer._weighted_aggregate(client_updates, data_sizes)`
  - 按数据量权重聚合客户端更新。
- `FLServer._aggregate_stats(stats, method)`
  - 统计中位数/分位数/裁剪比例（可附加 mean/std）。
- `FLServer.update_global_model(global_update, stats_aggregated)`
  - 更新参数并更新 `clip_norm`。
- `FLServer._update_clip_norm(stats)`
  - `ADAPTIVE`：按裁剪比例增减阈值。
  - `EMA`：按统计中位数做指数滑动平均。
- `FLServer.prepare_broadcast()`
  - 下发策略：全量参数或 Top-k 稀疏下发。
- `FLServer.evaluate(test_loader)`
  - 测试集评估，返回 loss/accuracy/correct/total。

### `edge/__init__.py`

作用：导出 `TrustedEdge`、`SimpleAggregator`、`EdgeOutput`。

### `edge/aggregator.py`

作用：可选边缘侧聚合器实现（与服务端逻辑对齐）。

类与函数块：

- `EdgeOutput`（dataclass）
  - 边缘聚合结果结构。
- `TrustedEdge.__init__(config)`
  - 初始化边缘配置。
- `TrustedEdge.aggregate(client_updates, data_sizes, stats)`
  - 聚合入口。
- `TrustedEdge._weighted_aggregate(...)`
  - 数据量加权聚合。
- `TrustedEdge._aggregate_stats(stats)`
  - 聚合统计量（可按 `StatsAggMethod.ALL` 输出 mean/std）。
- `SimpleAggregator.fedavg_aggregate(client_updates, data_sizes)`
  - 简化版 FedAvg 聚合工具。

---

## 5.5 模型模块

### `models/__init__.py`

作用：导出 `ResNet18`。

### `models/resnet.py`

作用：ResNet-18（GroupNorm 版本），适配 CIFAR/MNIST 统一输入流程。

函数块：

- `gn(num_channels)`
  - 统一 GroupNorm 构造函数。
- `BasicBlock.__init__(in_channels, out_channels, stride)`
  - 基本残差块：两层 3x3 卷积 + 可选 shortcut 对齐。
- `BasicBlock.forward(x)`
  - 残差前向。
- `ResNet18.__init__(num_classes)`
  - 构建 stem + 4 个 stage + 池化 + 分类头。
- `ResNet18._make_layer(in_channels, out_channels, stride)`
  - 每个 stage 由 2 个 BasicBlock 组成。
- `ResNet18.forward(x)`
  - 前向传播输出分类 logits。

---

## 5.6 数据模块

### `data/__init__.py`

作用：导出 `get_dataloader`。

### `data/utils.py`

作用：数据集下载、预处理、IID/Non-IID 切分。

函数块：

- `get_dataloader(num_clients, batch_size, alpha, iid, dataset, data_dir)`
  - 主入口：返回客户端训练加载器列表和全局测试加载器。
- `_build_dataset(dataset, data_dir)`
  - 构建 `CIFAR10` 或 `MNIST`（MNIST 转 3 通道 32x32）。
- `_split_iid(dataset, num_clients)`
  - 随机均匀切分。
- `_split_noniid_dirichlet(dataset, num_clients, alpha)`
  - Dirichlet 非 IID 切分。
- `_extract_targets(dataset)`
  - 统一抽取标签数组（list/tensor 兼容）。

---

## 5.7 压缩模块

### `compression/__init__.py`

作用：导出重要性评估、Top-k、残差器与统一压缩 API。

### `compression/topk.py`

作用：重要性计算 + Top-k 掩码 + 残差反馈。

函数块：

- `compute_fisher_information(model, dataloader, device, num_batches)`
  - 估计对角 Fisher 信息。
- `compute_importance_fisher_grad(gradient, fisher)`
  - `importance = fisher * |grad|`。
- `compute_importance_grad_squared(gradient)`
  - `importance = |grad|^2`。
- `compute_importance_grad_normalized(gradient)`
  - 每层 min-max 归一化的重要性。
- `topk_global(importance, k_ratio)`
  - 全局阈值 Top-k。
- `topk_layer(importance, k_ratio)`
  - 分层独立 Top-k。
- `topk_layer_norm_global(importance, k_ratio)`
  - 层内归一化后再做全局 Top-k。
- `topk_weighted_layer_norm(importance, k_ratio, weight_method)`
  - 层归一化 + 层权重（mean/median/sum/trimmed mean）后全局 Top-k。
- `ResidualAccumulator.__init__()`
  - 初始化残差缓存。
- `ResidualAccumulator.accumulate(gradient)`
  - 当前梯度叠加历史残差。
- `ResidualAccumulator.update(gradient, mask)`
  - 未上传分量写回残差缓存。
- `ResidualAccumulator.reset()`
  - 清空残差缓存。
- `compress_gradient(gradient, importance_strategy, topk_strategy, k_ratio, ...)`
  - 统一入口：重要性计算 + 掩码生成 + 稀疏化输出。

---

## 5.8 差分隐私模块

### `dp/__init__.py`

作用：导出会计器和噪声函数。

### `dp/noise.py`

作用：RDP 会计与噪声注入核心逻辑。

函数块：

- `_rdp_subsampled_gaussian(q, noise_multiplier, alpha)`
  - 泊松子采样高斯机制单步 RDP 近似。
- `_epsilon_from_rdp(rdp_values, delta)`
  - 从多阶 RDP 转换 `(epsilon, delta)`。
- `RDPAccountant.__post_init__()`
  - 初始化每轮目标与累计 RDP 容器。
- `RDPAccountant.current_epsilon()`
  - 当前累计隐私开销。
- `RDPAccountant.remaining_budget()`
  - 剩余预算。
- `RDPAccountant.is_exhausted()`
  - 是否耗尽预算。
- `RDPAccountant.solve_noise_multiplier_for_round(...)`
  - 二分反解本轮满足目标 epsilon 的 `z`。
- `RDPAccountant.solve_sigma_for_round(...)`
  - 兼容别名（返回同一含义）。
- `RDPAccountant.consume_round(noise_multiplier, q, steps, sigma)`
  - 消费一轮并更新累计 epsilon。
- `add_noise_to_tensor(tensor, sigma, generator)`
  - 各向同性高斯加噪。
- `compute_weighted_sensitivity(client_weights, clip_norm)`
  - 加权聚合在 add/remove 邻接下的敏感度上界。
- `allocate_client_noise_stds(client_weights, sigma_agg, strategy, ...)`
  - 从聚合目标方差反解客户端基准噪声标准差。
- `allocate_relative_scales(importance, mask, ...)`
  - 计算“相对噪声比例”（重要参数更小噪声）。
- `normalize_relative_scales(scales, mask)`
  - 归一化相对比例，保持方差预算一致。
- `add_heterogeneous_noise(tensor, sigma_base, relative_scales, mask, generator)`
  - 按元素异构标准差加噪。
- `PrivacyAccountant = RDPAccountant`
  - 向后兼容别名。

---

## 6. 输出结果文件说明

每次训练会生成目录（示例）：

- `checkpoints/<exp>_<timestamp>/`
- `logs/<exp>_<timestamp>/`
- `results/<exp>_<timestamp>/`（消融/超参脚本常用）

关键文件：

- `history.json`
  - 轮次级曲线数据：`test_accuracy/test_loss/clip_history/privacy_spent/upload_ratio/noise_multiplier/sigma_agg...`
- `experiment_record.json`
  - 完整记录：`experiment_info + config + summary + history`
- `config_summary.txt`
  - 纯配置快照
- `best_model.pt` / `final_model.pt` / `checkpoint_round_*.pt`
  - 模型权重与轮次状态

---

## 7. 关键参数速查

- 训练入口参数：
  - `--experiment`: `fedavg | dp_fedavg | dp_fedsam | proposed`
  - `--dataset`: `cifar10 | mnist`
  - `--num_rounds --num_clients --clients_per_round --alpha --iid --seed`
- DP 相关：
  - `--epsilon`
  - `--client_noise_allocation`: `uniform | heterogeneous`
  - `--client_variance_max_scale`
  - `--account_for_topk_in_q`
- 压缩相关：
  - `--topk_ratio`
- 可视化相关：
  - `--importance_viz --importance_viz_interval --importance_viz_client --importance_viz_max_elements`

