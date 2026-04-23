# 联邦学习实验方案与并行运行文档

本文档基于当前项目代码编写，适用于服务器上用单张 GPU 并行启动多个实验进程。当前训练主入口是 `train.py`，批量并行入口是 `run_parallel_experiments.py`。

## 1. 关键结论

当前项目有两类并行：

```text
实验级并行：run_parallel_experiments.py 同时启动多个 python train.py 子进程
客户端级并行：单个 train.py 内部用 ThreadPoolExecutor 并行训练多个客户端
```

两类参数不要混淆：

```text
--max-parallel
  这是 run_parallel_experiments.py 的参数，控制同时跑几个不同实验进程。

--intra-workers
  这是 run_parallel_experiments.py 的参数，会被转成 train.py 的 trainer.max_workers。

trainer.max_workers
  这是 train.py 的 OmegaConf 参数，控制单个实验内部每轮最多几个客户端线程并行。
```

推荐 5090 初始设置：

```bash
--max-parallel 3 --intra-workers 2
```

如果使用 `resnet18`，先保守使用：

```bash
--max-parallel 2 --intra-workers 1
```

如果用 `simple_cnn` 且 `nvidia-smi` 显示显存和 CPU 都有余量，可以试：

```bash
--max-parallel 4 --intra-workers 2
```

## 2. 当前代码入口说明

### 2.1 单次训练入口

`train.py` 使用 OmegaConf dotlist 参数，不是传统 argparse。

正确写法：

```bash
python train.py experiment.name=dp_fedavg dp.epsilon_total=8 server.initial_clip=1.0
```

错误写法：

```bash
python train.py --experiment dp_fedavg --epsilon 8
```

当前 `experiment.name` 的特殊逻辑：

```text
experiment.name=fedavg
  关闭 DP，关闭压缩，标准 FedAvg。

experiment.name=dp_fedavg
  开启 DP，关闭压缩，固定裁剪，标准本地训练。

experiment.name=dp_fedsam
  开启 DP，关闭压缩，固定裁剪，本地训练策略为 dp_fedsam。

experiment.name=proposed
  使用 configs/base.yaml 默认全模块配置，除非命令行显式覆盖。
```

### 2.2 并行实验入口

`run_parallel_experiments.py` 是外层调度器，会生成多个不同的 `python train.py ...` 命令，并用 `subprocess.Popen` 同时启动。

核心行为：

```text
1. 根据 --suite 生成实验矩阵。
2. 检查实验名是否重复。
3. 检查最终有效配置是否重复。
4. 同时启动最多 --max-parallel 个 train.py 子进程。
5. 某个进程结束后，从 pending 队列补上下一个实验。
```

它不是在一个 Python 进程里循环训练，而是操作系统层面的多进程并行：

```text
run_parallel_experiments.py
├── python train.py 实验 A
├── python train.py 实验 B
└── python train.py 实验 C
```

## 3. 并行如何保证实验参数不同

`run_parallel_experiments.py` 生成的每个 `Experiment` 都包含：

```text
name
  实验名，例如 val_dp_fedavg_clip0.5_eps1_s42。

overrides
  传给 train.py 的 OmegaConf 参数，例如 dp.epsilon_total=1 server.initial_clip=0.5。
```

验证实验中的示例：

```text
val_dp_fedavg_clip0.5_eps1_s42
  experiment.name=dp_fedavg
  dp.epsilon_total=1
  server.initial_clip=0.5

val_dp_fedavg_clip0.5_eps2_s42
  experiment.name=dp_fedavg
  dp.epsilon_total=2
  server.initial_clip=0.5

val_dp_fedavg_clip1_eps1_s42
  experiment.name=dp_fedavg
  dp.epsilon_total=1
  server.initial_clip=1.0

val_ours_core_k0.1_q0.7_s42
  experiment.name=proposed
  compressor.topk_ratio=0.1
  server.target_quantile=0.7

val_ours_full_fedprox_fisher_grad_s42
  experiment.name=proposed
  client.training_strategy=fedprox
  compressor.importance_strategy=fisher_grad
```

调度器启动前会做去重检查。检查规则模拟 OmegaConf 的“后写覆盖前写”：

```text
如果两个实验最终有效参数完全一样，即使实验名不同，也直接报错，不会启动。
```

因此 `--max-parallel 3` 的含义不是“把同一个实验跑 3 次”，而是“从不同配置的实验队列里同时取 3 个启动”。

## 4. 实验设计

### 4.1 验证实验

目的：选择各类方法的最佳参数，不作为最终主结果。

建议轮数：

```text
50 到 80 轮
```

包含：

```text
FedAvg
  无 DP，无压缩。作为非隐私上界。

DP-FedAvg
  clip in {0.5, 1.0, 1.5}
  epsilon in {1, 2, 4, 6, 8, 10}

ours-core
  只验证核心结构，不启用最复杂策略。
  topk in {0.05, 0.1, 0.2}
  target_quantile in {0.5, 0.7}

ours-full
  在 ours-core 较优配置上启用残差、正则、全局 importance、异构噪声、importance freeze。
  regularizer in {contrastive, fedprox}
  importance in {grad_normalized, fisher_grad}

DP-FedSAM
  可选。如果时间紧，可以只在主实验中加。
```

### 4.2 主对比实验

目的：基于验证实验选出的最优参数，比较不同隐私预算下的性能。

建议轮数：

```text
100 到 200 轮
```

建议隐私预算：

```text
epsilon in {1, 2, 4, 6, 8, 10}
```

方法：

```text
FedAvg
DP-FedAvg，使用验证实验选出的 best_dp_clip
ours-full，使用验证实验选出的 best_topk、best_regularizer、best_importance 等
DP-FedSAM，可选
```

建议至少跑一个 seed：

```text
seed = 42
```

论文结果更稳妥的设置：

```text
seed in {42, 43, 44}
```

### 4.3 粗粒度消融实验

目的：回答大模块是否有用。

建议固定：

```text
epsilon = 8
使用主实验选出的 ours-full 最优参数
```

变体：

```text
full
  完整方法。

no_adaptive_clip
  关闭自适应裁剪，改为固定裁剪。

no_compression
  关闭 Top-k 压缩和残差。

no_heterogeneous_noise
  关闭异构噪声，保留其他模块。
```

### 4.4 细粒度消融实验

目的：在确认大模块有效后，进一步验证具体策略选择。

建议固定：

```text
epsilon = 8
数据集、轮数、客户端数、采样率与主实验一致
```

变体：

```text
regularizer_standard
regularizer_fedprox
regularizer_contrastive

importance_grad_normalized
importance_grad_squared
importance_fisher_grad

topk_global
topk_layer
topk_layer_norm_global
topk_weighted_layer_norm

no_residual
no_global_prior
no_importance_freeze

mix_lambda_0
mix_lambda_02
mix_lambda_04
mix_lambda_06
```

### 4.5 超参数敏感度实验

目的：证明方法对关键超参数不脆弱。

建议：

```text
topk in {0.05, 0.1, 0.2, 0.3}
alpha in {0.1, 0.3, 0.5, 1.0}
lr in {0.003, 0.01, 0.03}
```

如果时间足够，可以额外加入：

```text
server.target_quantile in {0.5, 0.7, 0.9}
server.clip_lr in {0.1, 0.2, 0.4}
dp.global_local_mix_lambda in {0.0, 0.2, 0.4, 0.6}
```

## 5. 服务器运行前检查

进入项目目录：

```bash
cd /path/to/experiment
```

确认 GPU 可见：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

如果输出 `cuda: False`，说明当前环境不是 CUDA 版 PyTorch，必须先安装 GPU 版 PyTorch。

检查语法：

```bash
python -m py_compile train.py run_parallel_experiments.py
```

先 dry-run，不启动训练：

```bash
python run_parallel_experiments.py --suite validation --dry-run --rounds 1
```

dry-run 的作用：

```text
打印所有即将启动的 train.py 命令。
检查实验参数是否互不相同。
检查实验名和有效配置是否重复。
```

## 6. 推荐运行命令

### 6.1 验证实验

```bash
cd /path/to/experiment

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

nohup python -u run_parallel_experiments.py \
  --suite validation \
  --max-parallel 3 \
  --intra-workers 2 \
  --rounds 60 \
  --model simple_cnn \
  --dataset cifar10 \
  --seeds 42 \
  --validation-epsilons 1,2,4,6,8,10 \
  --dp-clips 0.5,1.0,1.5 \
  --core-topks 0.05,0.1,0.2 \
  --core-target-quantiles 0.5,0.7 \
  --include-fedsam \
  > validation_launcher.log 2>&1 &
```

注释：

```text
--suite validation
  使用验证实验矩阵。

--max-parallel 3
  同时最多跑 3 个不同的 train.py 实验进程。

--intra-workers 2
  每个 train.py 内部 trainer.max_workers=2。

--rounds 60
  验证阶段不用跑太久，先用 60 轮选参数。

--validation-epsilons 1,2,4,6,8,10
  DP-FedAvg 验证用的隐私预算。

--dp-clips 0.5,1.0,1.5
  DP-FedAvg 固定裁剪阈值搜索范围。

--core-topks 0.05,0.1,0.2
  ours-core 的 Top-k 压缩率搜索范围。

--core-target-quantiles 0.5,0.7
  自适应裁剪目标分位数搜索范围。

--include-fedsam
  同时跑 DP-FedSAM 验证实验。
```

### 6.2 主对比实验

假设验证实验选出：

```text
best_dp_clip = 1.0
best_topk = 0.1
best_regularizer = contrastive
best_importance = fisher_grad
best_target_quantile = 0.7
best_clip_lr = 0.2
best_mix_lambda = 0.4
```

运行：

```bash
nohup python -u run_parallel_experiments.py \
  --suite main \
  --max-parallel 3 \
  --intra-workers 2 \
  --rounds 100 \
  --model simple_cnn \
  --dataset cifar10 \
  --seeds 42 \
  --main-epsilons 1,2,4,6,8,10 \
  --best-dp-clip 1.0 \
  --best-topk 0.1 \
  --best-regularizer contrastive \
  --best-importance fisher_grad \
  --best-target-quantile 0.7 \
  --best-clip-lr 0.2 \
  --best-mix-lambda 0.4 \
  --include-fedsam \
  > main_launcher.log 2>&1 &
```

如果要跑 3 个 seed：

```bash
nohup python -u run_parallel_experiments.py \
  --suite main \
  --max-parallel 3 \
  --intra-workers 2 \
  --rounds 100 \
  --model simple_cnn \
  --dataset cifar10 \
  --seeds 42,43,44 \
  --main-epsilons 1,2,4,6,8,10 \
  --best-dp-clip 1.0 \
  --best-topk 0.1 \
  --best-regularizer contrastive \
  --best-importance fisher_grad \
  --include-fedsam \
  > main_3seeds_launcher.log 2>&1 &
```

### 6.3 粗粒度消融实验

```bash
nohup python -u run_parallel_experiments.py \
  --suite ablation_macro \
  --max-parallel 3 \
  --intra-workers 2 \
  --rounds 100 \
  --model simple_cnn \
  --dataset cifar10 \
  --seeds 42 \
  --base-epsilon 8 \
  --best-dp-clip 1.0 \
  --best-topk 0.1 \
  --best-regularizer contrastive \
  --best-importance fisher_grad \
  > ablation_macro_launcher.log 2>&1 &
```

### 6.4 细粒度消融实验

```bash
nohup python -u run_parallel_experiments.py \
  --suite ablation_fine \
  --max-parallel 3 \
  --intra-workers 2 \
  --rounds 100 \
  --model simple_cnn \
  --dataset cifar10 \
  --seeds 42 \
  --base-epsilon 8 \
  --best-topk 0.1 \
  --best-regularizer contrastive \
  --best-importance fisher_grad \
  > ablation_fine_launcher.log 2>&1 &
```

### 6.5 超参数敏感度实验

```bash
nohup python -u run_parallel_experiments.py \
  --suite sensitivity \
  --max-parallel 3 \
  --intra-workers 2 \
  --rounds 80 \
  --model simple_cnn \
  --dataset cifar10 \
  --seeds 42 \
  --base-epsilon 8 \
  --sens-topks 0.05,0.1,0.2,0.3 \
  --sens-alphas 0.1,0.3,0.5,1.0 \
  --sens-lrs 0.003,0.01,0.03 \
  > sensitivity_launcher.log 2>&1 &
```

## 7. 单条训练命令示例

### 7.1 FedAvg

```bash
python -u train.py \
  experiment.name=fedavg \
  trainer.num_rounds=100 \
  trainer.max_workers=2 \
  data.dataset=cifar10 \
  model.name=simple_cnn
```

### 7.2 DP-FedAvg

```bash
python -u train.py \
  experiment.name=dp_fedavg \
  dp.epsilon_total=8 \
  server.initial_clip=1.0 \
  trainer.num_rounds=100 \
  trainer.max_workers=2 \
  data.dataset=cifar10 \
  model.name=simple_cnn
```

### 7.3 DP-FedSAM

```bash
python -u train.py \
  experiment.name=dp_fedsam \
  dp.epsilon_total=8 \
  server.initial_clip=1.0 \
  trainer.num_rounds=100 \
  trainer.max_workers=2 \
  data.dataset=cifar10 \
  model.name=simple_cnn
```

### 7.4 ours-full

```bash
python -u train.py \
  experiment.name=proposed \
  dp.epsilon_total=8 \
  client.training_strategy=contrastive \
  compressor.type=topk \
  compressor.topk_ratio=0.1 \
  compressor.use_residual=true \
  compressor.importance_strategy=fisher_grad \
  compressor.topk_strategy=weighted_layer_norm \
  server.clip_update_method=adaptive \
  server.target_quantile=0.7 \
  server.clip_lr=0.2 \
  dp.use_heterogeneous_noise=true \
  dp.use_global_importance_for_topk=true \
  dp.global_local_mix_lambda=0.4 \
  dp.enable_importance_freeze=true \
  trainer.num_rounds=100 \
  trainer.max_workers=2 \
  data.dataset=cifar10 \
  model.name=simple_cnn
```

## 8. 监控命令

查看 GPU：

```bash
watch -n 1 nvidia-smi
```

查看 launcher 日志：

```bash
tail -f validation_launcher.log
```

查看某个具体实验日志：

```bash
tail -f runs/launcher_logs/*/*.out
```

查看正在运行的训练进程：

```bash
ps -ef | grep "python.*train.py" | grep -v grep
```

查看进程数量：

```bash
ps -ef | grep "python.*train.py" | grep -v grep | wc -l
```

## 9. 输出目录

`run_parallel_experiments.py` 默认输出到：

```text
runs/
```

具体结构：

```text
runs/
├── checkpoints/
│   ├── 20260423_120000_main_dp_fedavg_eps1_s42/
│   └── 20260423_120000_main_ours_full_eps1_s42/
├── logs/
│   ├── 20260423_120000_main_dp_fedavg_eps1_s42/
│   │   ├── experiment_record.json
│   │   └── history.json
│   └── 20260423_120000_main_ours_full_eps1_s42/
│       ├── experiment_record.json
│       └── history.json
└── launcher_logs/
    └── 20260423_120000/
        ├── main_dp_fedavg_eps1_s42.out
        └── main_ours_full_eps1_s42.out
```

重点文件：

```text
experiment_record.json
  包含完整配置、summary 和历史曲线。

history.json
  包含每轮 accuracy、loss、clip、epsilon、SNR、upload ratio 等曲线。

*.out
  对应每个子进程的 stdout/stderr。
```

## 10. 如何根据验证结果选参数

每个 `experiment_record.json` 里有：

```json
{
  "summary": {
    "best_accuracy": 0.0,
    "final_accuracy": 0.0,
    "final_loss": 0.0,
    "final_epsilon": 0.0,
    "final_clip_norm": 0.0
  }
}
```

选择规则建议：

```text
DP-FedAvg
  对每个 epsilon，选择 best_accuracy 最高的 clip。
  如果不同 clip 精度接近，优先选择曲线更稳定、final_accuracy 更高的 clip。

ours-core
  选择 best_accuracy 高且 upload_ratio 较低、SNR 不异常的 topk/target_quantile。

ours-full
  选择 best_accuracy 和 final_accuracy 同时较高的 regularizer/importance。
  如果 fisher_grad 很慢但只提升很小，可以优先 grad_normalized。
```

建议不要只看 `best_accuracy`：

```text
best_accuracy
  反映峰值表现。

final_accuracy
  反映训练结束稳定性。

test_loss
  辅助判断是否过拟合或不稳定。

clip_history
  判断自适应裁剪是否发散。

privacy_spent
  确认 epsilon 是否符合预算。

upload_ratio
  验证压缩是否真的生效。
```

## 11. 常见问题

### 11.1 为什么 GPU 利用率还是不高

可能原因：

```text
模型太小，simple_cnn 单个 batch 计算量有限。
每个客户端数据少，kernel 调用碎片化。
Python 调度和 DataLoader 成为瓶颈。
DP 噪声、压缩、聚合有 CPU/GPU 同步开销。
```

处理：

```text
先把 --max-parallel 从 2 调到 3，再到 4。
保持 --intra-workers 1 或 2，不要盲目调太大。
观察单位时间完成 round 数，而不是只看 GPU 利用率。
```

### 11.2 为什么不能无限增加 --max-parallel

原因：

```text
多个进程会同时占用显存。
多个进程会争抢 GPU 上下文。
CPU 数据加载和 Python 调度可能先打满。
磁盘日志写入也会增加开销。
```

推荐：

```text
simple_cnn: 3 到 4 个进程
resnet18: 1 到 2 个进程
```

### 11.3 --max-parallel 和 --intra-workers 怎么配

建议组合：

```text
保守测试:
  --max-parallel 2 --intra-workers 1

推荐起点:
  --max-parallel 3 --intra-workers 2

simple_cnn 激进:
  --max-parallel 4 --intra-workers 2

resnet18:
  --max-parallel 2 --intra-workers 1
```

### 11.4 如何确认跑的不是重复实验

先 dry-run：

```bash
python run_parallel_experiments.py --suite validation --dry-run
```

检查输出命令中的关键参数是否不同：

```text
experiment.name
dp.epsilon_total
server.initial_clip
compressor.topk_ratio
client.training_strategy
compressor.importance_strategy
server.target_quantile
seed
```

如果有效配置重复，脚本会直接报错。

### 11.5 Kaggle P100 上 15% 利用率，5090 应该怎么跑

先用短轮数压测：

```bash
python run_parallel_experiments.py --suite validation --max-parallel 2 --intra-workers 2 --rounds 10
python run_parallel_experiments.py --suite validation --max-parallel 3 --intra-workers 2 --rounds 10
python run_parallel_experiments.py --suite validation --max-parallel 4 --intra-workers 2 --rounds 10
```

比较：

```text
总 wall time
每个实验平均 round time
GPU 利用率
显存占用
CPU 利用率
是否出现 OOM
```

选择单位时间完成 round 数最高的组合。

## 12. 建议执行顺序

推荐按下面顺序执行：

```text
1. py_compile 检查。
2. validation dry-run。
3. validation 短跑 10 轮，测试并行度。
4. validation 正式跑 60 轮。
5. 根据验证结果确定 best_dp_clip、best_topk、best_regularizer、best_importance。
6. main 正式跑 100 轮。
7. ablation_macro。
8. ablation_fine。
9. sensitivity。
10. 如果时间允许，对 main 和关键消融补 3 seeds。
```

最小可接受实验集：

```text
validation: seed 42, 60 轮
main: seed 42, 100 轮
ablation_macro: seed 42, 100 轮
sensitivity: seed 42, 80 轮
```

论文更稳妥实验集：

```text
main: seeds 42,43,44
ablation_macro: seeds 42,43,44
关键 sensitivity: seeds 42,43,44
```

