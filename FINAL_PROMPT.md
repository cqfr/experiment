# 最终提示词（可直接用于代码生成/实验执行）

你是联邦学习（FL）+ 差分隐私（DP）实验工程师。请在同一套可配置代码框架中，实现并公平对比以下四种方法，并保证实验可复现。

## 1. 目标与范围

在 `ResNet-18` + 图像分类任务上完成：

1. `FedAvg`（无DP）
2. `DP-FedAvg`
3. `DP-FedSAM`
4. `Ours`（对比学习正则 + 残差累积 + 重要性感知Top-k + 自适应裁剪 + 异构噪声）

数据集至少覆盖：

- `CIFAR-10`
- `MNIST`
- `cifar-100`

## 2. 统一建模符号

客户端上传统一定义为：

- `Delta_w_i = w_i_local - w_t_global`

服务端更新统一定义为：

- `w_{t+1} = w_t + eta * sum_i (|D_i| / sum_j |D_j|) * Delta_w_i`

## 3. Baseline 定义

### 3.1 FedAvg

- 客户端本地训练后上传 `Delta_w_i`
- 服务端按数据量加权聚合并更新

### 3.2 DP-FedAvg

- 在 `FedAvg` 基础上：
  - 客户端上传前执行 L2 裁剪
  - 对裁剪后更新添加同方差高斯噪声

### 3.3 DP-FedSAM

- 在 `DP-FedAvg` 基础上将本地优化器替换为 `SAM` 风格更新（两步法）

## 4. Ours 方法（客户端与服务端流程）

### 客户端流程

1. 接收 `w_t_global`
2. 本地训练：
   - `L_total = L_CE + L_contrastive`
   - `L_contrastive = alpha * ||w - w_g||^2 + beta * max(0, m - ||w - w_old||)^2`
   - 注意：使用有界形式，避免发散
3. 残差累积：`Delta_w = Delta_w_local + residual_{t-1}`
4. 参数重要性评估（实现可切换）：
   - `Fisher * |Delta_w|`
   - `|Delta_w|^2`
   - `|Delta_w|` 层内归一化
5. Top-k 压缩（实现可切换）：
   - 全局Top-k
   - 层级Top-k
   - 层级归一化+全局Top-k
   - 加权层级归一化+全局Top-k
6. 裁剪：按当前 `C_t` 对稀疏更新进行 L2 裁剪
7. 异构噪声：
   - 先由每轮目标隐私预算反解 `sigma_base`
   - 再按重要性给出“相对噪声比例”
   - 最后统一缩放，使该轮隐私约束满足
8. 上传：
   - `Delta_w_noisy`
   - `|D_i|`
   - `stat_i`（`clipped_flag` 或 `L2_norm`）

### 服务端流程

9. 收集客户端上传
10. 按数据量加权聚合
11. 统计处理（可切换）：
   - 分位数统计
   - 全量统计
12. 更新全局模型
13. 更新裁剪阈值（可切换）：
   - Adaptive clipping
   - EMA
14. 下发（可切换）：
   - 全量模型
   - Top-k 下发

## 5. 隐私会计要求（关键）

必须使用 `RDP accountant` 做逐轮核算：

- 给定每轮目标 `epsilon_round`、`delta`、采样率 `q`、机制步数 `steps`，反解该轮 `sigma_base`
- 每轮训练后更新累计隐私消耗，输出 `epsilon_spent(t)`

说明：

- 假设服务器可信，客户端上传统计量 `stat_i` 允许用于裁剪阈值更新（不额外做保护）

## 6. 工程要求

- 配置驱动（dataclass/YAML/JSON 皆可）
- 训练日志必须包含：
  - `accuracy`
  - `loss`
  - `epsilon_spent`
  - `clip_norm`
  - `upload_ratio` / 通信压缩率
- 输出 `history.json` 与 `experiment_record.json`
- 支持固定随机种子复现

## 7. 实验输出要求

必须给出以下对比曲线：

1. `test accuracy vs round`
2. `test loss vs round`
3. `epsilon_spent vs round`
4. `clip_norm vs round`
5. `upload_ratio vs round`

## 8. 消融实验要求

至少包含：

1. 完整方法
2. 去掉压缩
3. 去掉自适应裁剪
4. 去掉异构噪声
5. 去掉对比正则

并和 `FedAvg / DP-FedAvg / DP-FedSAM` 一并对比。
