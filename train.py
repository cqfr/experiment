"""
train.py
联邦学习主训练循环
整合 Client、Edge、Server 实现完整训练流程
"""

import os
import copy
import random
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm

from config import ExperimentConfig, get_fedavg_config, get_dp_fedavg_config, get_proposed_config
from models.resnet import ResNet18
from data.utils import get_dataloader
from client.client import FLClient, ClientUpdate
from server.server import FLServer
from dp.noise import PrivacyAccountant


class FLTrainer:
    """
    联邦学习训练器
    
    管理完整的训练流程：
    1. 初始化所有组件
    2. 每轮选择客户端
    3. 客户端本地训练
    4. Edge 聚合
    5. Server 更新
    6. 评估和日志
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载数据
        print("加载数据...")
        self.train_loaders, self.test_loader = get_dataloader(
            num_clients=config.num_clients,
            batch_size=config.batch_size,
            alpha=config.alpha,
            iid=config.iid,
        )
        print(f"  客户端数: {config.num_clients}")
        print(f"  每个客户端平均样本数: {sum(len(l.dataset) for l in self.train_loaders) // config.num_clients}")
        print(f"  测试集样本数: {len(self.test_loader.dataset)}")
        
        # 创建模型
        print("创建模型...")
        self.model = ResNet18(num_classes=config.num_classes).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  模型参数量: {total_params / 1e6:.2f}M")
        
        # 创建服务器
        self.server = FLServer(
            model=self.model,
            config=config.server,
            device=self.device,
        )
        
        # 创建所有客户端
        print("初始化客户端...")
        self.clients: List[FLClient] = []
        for i in range(config.num_clients):
            client = FLClient(
                client_id=i,
                model=self.model,
                dataloader=self.train_loaders[i],
                config=config.client,
                dp_config=config.dp,
                device=self.device,
            )
            self.clients.append(client)
        
        # 隐私记账器
        if config.dp.enabled:
            self.privacy_accountant = PrivacyAccountant(
                epsilon_total=config.dp.epsilon_total,
                delta=config.dp.delta,
                num_rounds=config.num_rounds,
            )
            print(f"  DP 隐私预算: ε={config.dp.epsilon_total}, δ={config.dp.delta}")
            print(f"  每轮预算: ε={self.privacy_accountant.epsilon_per_round:.4f}")
        else:
            self.privacy_accountant = None
        
        # 训练历史
        self.history = {
            "rounds": [],
            "train_metrics": [],
            "test_metrics": [],
            "clip_history": [],
            "privacy_spent": [],
        }
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _select_clients(self, num_clients: int) -> List[int]:
        """随机选择参与本轮的客户端"""
        return random.sample(range(self.config.num_clients), num_clients)
    
    def train_round(self, round_num: int) -> Dict:
        """
        执行一轮联邦学习训练
        """
        # 获取当前裁剪阈值和隐私预算
        clip_norm = self.server.get_clip_norm()
        
        if self.privacy_accountant is not None:
            epsilon_round = self.privacy_accountant.get_round_epsilon()
        else:
            epsilon_round = float('inf')
        
        # 选择客户端
        selected_ids = self._select_clients(self.config.clients_per_round)
        sampling_rate = self.config.clients_per_round / self.config.num_clients
        
        # 获取全局模型
        global_weights = self.server.get_global_weights()
        
        # 客户端训练
        client_updates: List[Dict[str, torch.Tensor]] = []
        data_sizes: List[int] = []
        stats: List[float] = []
        noise_sigmas: List[float] = []
        
        for client_id in selected_ids:
            client = self.clients[client_id]
            
            update = client.train_and_upload(
                global_weights=global_weights,
                clip_norm=clip_norm,
                epsilon_round=epsilon_round,
                sampling_rate=sampling_rate,
            )
            
            client_updates.append(update.gradient)
            data_sizes.append(update.data_size)
            stats.append(update.stat)
            noise_sigmas.append(update.noise_sigma)
        
        # Server 聚合（数据量加权）
        aggregated = self.server.aggregate(client_updates, data_sizes, stats)
        
        # Server 更新全局模型
        train_metrics = self.server.update_global_model(
            aggregated.global_gradient,
            aggregated.stats_aggregated,
        )
        
        # 记录平均噪声强度
        train_metrics["avg_noise_sigma"] = sum(noise_sigmas) / len(noise_sigmas) if noise_sigmas else 0.0
        
        # 更新隐私预算
        if self.privacy_accountant is not None:
            self.privacy_accountant.consume_round(epsilon_round)
            train_metrics["epsilon_spent"] = self.privacy_accountant.epsilon_spent
            train_metrics["epsilon_remaining"] = self.privacy_accountant.remaining_budget()
        
        return train_metrics    
    def evaluate(self) -> Dict:
        """评估全局模型"""
        return self.server.evaluate(self.test_loader)
    
    def train(self) -> Dict:
        """
        完整训练流程
        """
        print(f"\n{'='*60}")
        print(f"开始训练")
        print(f"{'='*60}")
        
        best_acc = 0.0
        
        for round_num in tqdm(range(1, self.config.num_rounds + 1), desc="训练进度"):
            # 检查隐私预算
            if self.privacy_accountant is not None and self.privacy_accountant.is_exhausted():
                print(f"\n隐私预算耗尽，在第 {round_num} 轮停止训练")
                break
            
            # 训练一轮
            train_metrics = self.train_round(round_num)
            
            # 评估
            test_metrics = self.evaluate()
            
            # 记录历史
            self.history["rounds"].append(round_num)
            self.history["train_metrics"].append(train_metrics)
            self.history["test_metrics"].append(test_metrics)
            self.history["clip_history"].append(train_metrics.get("new_clip", 0))
            
            if self.privacy_accountant is not None:
                self.history["privacy_spent"].append(train_metrics.get("epsilon_spent", 0))
            
            # 打印日志
            if round_num % self.config.log_interval == 0:
                log_msg = (
                    f"Round {round_num:3d} | "
                    f"Acc: {test_metrics['accuracy']:.2%} | "
                    f"Loss: {test_metrics['loss']:.4f} | "
                    f"Clip: {train_metrics.get('new_clip', 0):.3f}"
                )
                if self.privacy_accountant is not None:
                    log_msg += f" | ε: {train_metrics.get('epsilon_spent', 0):.2f}"
                    log_msg += f" | σ: {train_metrics.get('avg_noise_sigma', 0):.4f}"
                tqdm.write(log_msg)
            
            # 保存最佳模型
            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                self._save_checkpoint(round_num, is_best=True)
            
            # 定期保存
            if round_num % self.config.save_interval == 0:
                self._save_checkpoint(round_num, is_best=False)
        
        # 保存最终模型和历史
        self._save_checkpoint(self.config.num_rounds, is_best=False, is_final=True)
        self._save_history()
        
        print(f"\n{'='*60}")
        print(f"训练完成")
        print(f"最佳准确率: {best_acc:.2%}")
        if self.privacy_accountant is not None:
            print(f"总隐私消耗: ε = {self.privacy_accountant.epsilon_spent:.2f}")
        print(f"{'='*60}")
        
        return {
            "best_accuracy": best_acc,
            "final_accuracy": test_metrics["accuracy"],
            "final_loss": test_metrics["loss"],
            "total_rounds": len(self.history["rounds"]),
        }
    
    def _save_checkpoint(self, round_num: int, is_best: bool = False, is_final: bool = False):
        """保存检查点"""
        checkpoint = {
            "round": round_num,
            "model_state_dict": self.server.global_model.state_dict(),
            "clip_norm": self.server.clip_norm,
            "config": self.config.__dict__,
        }
        
        if is_best:
            path = os.path.join(self.config.save_dir, "best_model.pt")
        elif is_final:
            path = os.path.join(self.config.save_dir, "final_model.pt")
        else:
            path = os.path.join(self.config.save_dir, f"checkpoint_round_{round_num}.pt")
        
        torch.save(checkpoint, path)
    
    def _save_history(self):
        """保存训练历史和完整实验配置"""
        
        # 1. 将配置转换为可序列化格式
        config_serializable = self._config_to_dict(self.config)
        
        # 2. 训练历史数据
        history_data = {
            "rounds": self.history["rounds"],
            "test_accuracy": [m["accuracy"] for m in self.history["test_metrics"]],
            "test_loss": [m["loss"] for m in self.history["test_metrics"]],
            "clip_history": self.history["clip_history"],
            "privacy_spent": self.history["privacy_spent"],
            "noise_sigma": [m.get("avg_noise_sigma", 0.0) for m in self.history["train_metrics"]],
        }
        
        # 3. 汇总统计
        summary = {
            "best_accuracy": max(history_data["test_accuracy"]) if history_data["test_accuracy"] else 0.0,
            "final_accuracy": history_data["test_accuracy"][-1] if history_data["test_accuracy"] else 0.0,
            "final_loss": history_data["test_loss"][-1] if history_data["test_loss"] else 0.0,
            "total_rounds": len(history_data["rounds"]),
            "final_epsilon": history_data["privacy_spent"][-1] if history_data["privacy_spent"] else 0.0,
            "final_clip_norm": history_data["clip_history"][-1] if history_data["clip_history"] else 0.0,
        }
        
        # 4. 完整的实验记录
        full_record = {
            "experiment_info": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(self.device),
            },
            "config": config_serializable,
            "summary": summary,
            "history": history_data,
        }
        
        # 5. 保存完整记录
        path = os.path.join(self.config.log_dir, "experiment_record.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(full_record, f, indent=2, ensure_ascii=False)
        
        print(f"实验记录已保存到: {path}")
        
        # 6. 额外保存一份简洁的配置摘要（方便快速查看）
        legacy_history_path = os.path.join(self.config.log_dir, "history.json")
        with open(legacy_history_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        print(f"History saved to: {legacy_history_path}")
        config_summary_path = os.path.join(self.config.log_dir, "config_summary.txt")
        self._save_config_summary(config_summary_path)
        print(f"配置摘要已保存到: {config_summary_path}")
    
    def _config_to_dict(self, config: ExperimentConfig) -> dict:
        """将实验配置转换为可序列化的字典"""
        return {
            # 基础设置
            "seed": config.seed,
            "num_clients": config.num_clients,
            "clients_per_round": config.clients_per_round,
            "num_rounds": config.num_rounds,
            "batch_size": config.batch_size,
            
            # 数据集
            "dataset": config.dataset,
            "iid": config.iid,
            "alpha": config.alpha,
            
            # 模型
            "model": config.model,
            "num_classes": config.num_classes,
            
            # 客户端配置
            "client": {
                "training_strategy": config.client.training_strategy.value,
                "local_epochs": config.client.local_epochs,
                "lr": config.client.lr,
                "momentum": config.client.momentum,
                "weight_decay": config.client.weight_decay,
                "alpha": config.client.alpha,
                "beta": config.client.beta,
                "mu": config.client.mu,
                "importance_strategy": config.client.importance_strategy.value,
                "topk_strategy": config.client.topk_strategy.value,
                "topk_ratio": config.client.topk_ratio,
                "layer_weight_method": config.client.layer_weight_method.value,
                "stat_type": config.client.stat_type.value,
                "use_residual": config.client.use_residual,
            },
            
            # 服务器配置
            "server": {
                "server_lr": config.server.server_lr,
                "initial_clip": config.server.initial_clip,
                "clip_update_method": config.server.clip_update_method.value,
                "target_quantile": config.server.target_quantile,
                "clip_lr": config.server.clip_lr,
                "ema_alpha": config.server.ema_alpha,
                "downlink_strategy": config.server.downlink_strategy.value,
                "downlink_topk_ratio": config.server.downlink_topk_ratio,
            },
            
            # 差分隐私配置
            "dp": {
                "enabled": config.dp.enabled,
                "epsilon_total": config.dp.epsilon_total,
                "delta": config.dp.delta,
                "use_subsampling_amplification": config.dp.use_subsampling_amplification,
                "use_sparsity_amplification": config.dp.use_sparsity_amplification,
                "use_heterogeneous_noise": config.dp.use_heterogeneous_noise,
            },
            
            # 日志与保存
            "log_interval": config.log_interval,
            "save_interval": config.save_interval,
            "save_dir": config.save_dir,
            "log_dir": config.log_dir,
        }
    
    def _save_config_summary(self, path: str):
        """保存配置摘要（人类可读格式）"""
        c = self.config
        
        lines = [
            "=" * 60,
            "实验配置摘要",
            "=" * 60,
            "",
            "【基础设置】",
            f"  随机种子: {c.seed}",
            f"  客户端总数: {c.num_clients}",
            f"  每轮参与客户端: {c.clients_per_round}",
            f"  总轮数: {c.num_rounds}",
            f"  批大小: {c.batch_size}",
            "",
            "【数据集】",
            f"  数据集: {c.dataset}",
            f"  IID: {c.iid}",
            f"  Dirichlet α: {c.alpha}",
            "",
            "【模型】",
            f"  模型: {c.model}",
            f"  类别数: {c.num_classes}",
            "",
            "【客户端训练】",
            f"  训练策略: {c.client.training_strategy.value}",
            f"  本地训练轮数: {c.client.local_epochs}",
            f"  学习率: {c.client.lr}",
            f"  动量: {c.client.momentum}",
            f"  权重衰减: {c.client.weight_decay}",
        ]
        
        if c.client.training_strategy.value == "contrastive":
            lines.extend([
                f"  Contrastive α (全局一致性): {c.client.alpha}",
                f"  Contrastive β (局部多样性): {c.client.beta}",
            ])
        elif c.client.training_strategy.value == "fedprox":
            lines.append(f"  FedProx μ: {c.client.mu}")
        
        lines.extend([
            "",
            "【梯度压缩】",
            f"  重要性策略: {c.client.importance_strategy.value}",
            f"  Top-k 策略: {c.client.topk_strategy.value}",
            f"  压缩率 k: {c.client.topk_ratio:.1%}",
            f"  层权重方法: {c.client.layer_weight_method.value}",
            f"  残差累积: {c.client.use_residual}",
            "",
            "【服务器】",
            f"  全局学习率: {c.server.server_lr}",
            f"  初始裁剪阈值: {c.server.initial_clip}",
            f"  裁剪更新方法: {c.server.clip_update_method.value}",
            f"  目标分位数: {c.server.target_quantile}",
            f"  裁剪学习率: {c.server.clip_lr}",
            f"  EMA α: {c.server.ema_alpha}",
            "",
            "【差分隐私】",
            f"  启用 DP: {c.dp.enabled}",
        ])
        
        if c.dp.enabled:
            lines.extend([
                f"  总隐私预算 ε: {c.dp.epsilon_total}",
                f"  δ: {c.dp.delta}",
                f"  子采样放大: {c.dp.use_subsampling_amplification}",
                f"  稀疏化放大: {c.dp.use_sparsity_amplification}",
                f"  异构噪声: {c.dp.use_heterogeneous_noise}",
            ])
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def run_experiment(
    experiment_name: str,
    config: ExperimentConfig,
) -> Dict:
    """运行单个实验"""
    print(f"\n{'#'*60}")
    print(f"实验: {experiment_name}")
    print(f"{'#'*60}")
    
    # 更新保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.save_dir = f"./checkpoints/{experiment_name}_{timestamp}"
    config.log_dir = f"./logs/{experiment_name}_{timestamp}"
    
    # 创建训练器
    trainer = FLTrainer(config)
    
    # 训练
    results = trainer.train()
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="联邦学习训练")
    parser.add_argument("--experiment", type=str, default="proposed",
                        choices=["fedavg", "dp_fedavg", "proposed"],
                        help="实验类型")
    parser.add_argument("--num_rounds", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--num_clients", type=int, default=100,
                        help="客户端总数")
    parser.add_argument("--clients_per_round", type=int, default=10,
                        help="每轮参与的客户端数")
    parser.add_argument("--epsilon", type=float, default=8.0,
                        help="隐私预算 ε")
    parser.add_argument("--topk_ratio", type=float, default=0.1,
                        help="Top-k 压缩率")
    parser.add_argument("--iid", action="store_true",
                        help="使用 IID 数据划分")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet 分布参数（Non-IID 时使用）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 选择配置
    if args.experiment == "fedavg":
        config = get_fedavg_config()
    elif args.experiment == "dp_fedavg":
        config = get_dp_fedavg_config(epsilon=args.epsilon)
    else:
        config = get_proposed_config(epsilon=args.epsilon)
    
    # 更新配置
    config.num_rounds = args.num_rounds
    config.num_clients = args.num_clients
    config.clients_per_round = args.clients_per_round
    config.iid = args.iid
    config.alpha = args.alpha
    config.seed = args.seed
    config.client.topk_ratio = args.topk_ratio
    
    # 运行实验
    results = run_experiment(args.experiment, config)
    
    print(f"\n最终结果:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
