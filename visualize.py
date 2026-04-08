"""
visualize.py
实验结果可视化工具
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curves(
    history_paths: Dict[str, str],
    save_path: Optional[str] = None,
    metric: str = "accuracy",
):
    """
    绘制训练曲线对比图
    
    参数:
        history_paths: {实验名: history.json路径}
        save_path: 保存路径
        metric: "accuracy" 或 "loss"
    """
    plt.figure(figsize=(10, 6))
    
    for name, path in history_paths.items():
        with open(path, "r") as f:
            history = json.load(f)
        
        rounds = history["rounds"]
        
        if metric == "accuracy":
            values = history["test_accuracy"]
            ylabel = "Test Accuracy"
        else:
            values = history["test_loss"]
            ylabel = "Test Loss"
        
        plt.plot(rounds, values, label=name, linewidth=2)
    
    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"Training {metric.title()} Comparison", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()


def plot_privacy_accuracy_tradeoff(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
):
    """
    绘制隐私-精度权衡曲线
    
    参数:
        results: {实验名: {"epsilon": x, "accuracy": y}}
    """
    plt.figure(figsize=(8, 6))
    
    # 提取数据
    methods = list(results.keys())
    epsilons = [results[m].get("epsilon", 0) for m in methods]
    accuracies = [results[m]["best_accuracy"] for m in methods]
    
    # 按 epsilon 排序
    sorted_idx = np.argsort(epsilons)
    epsilons = [epsilons[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]
    methods = [methods[i] for i in sorted_idx]
    
    # 绘制
    plt.plot(epsilons, accuracies, 'o-', markersize=10, linewidth=2)
    
    # 标注
    for i, (eps, acc, name) in enumerate(zip(epsilons, accuracies, methods)):
        plt.annotate(
            f"{name}\n{acc:.1%}",
            (eps, acc),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
        )
    
    plt.xlabel("Privacy Budget ε", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("Privacy-Accuracy Tradeoff", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()


def plot_ablation_bar(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
):
    """
    绘制消融实验柱状图
    """
    plt.figure(figsize=(12, 6))
    
    names = list(results.keys())
    accuracies = [results[n]["best_accuracy"] * 100 for n in names]
    
    # 颜色
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    
    bars = plt.bar(range(len(names)), accuracies, color=colors)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha='center',
            va='bottom',
            fontsize=10,
        )
    
    plt.xticks(range(len(names)), names, rotation=45, ha='right', fontsize=10)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Ablation Study Results", fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()


def plot_sensitivity_curve(
    results: Dict[str, Dict],
    param_name: str,
    param_key: str,
    save_path: Optional[str] = None,
):
    """
    绘制超参数敏感性曲线
    
    参数:
        results: 实验结果字典
        param_name: 参数显示名称（如 "Compression Ratio k"）
        param_key: 结果中的参数键（如 "k"）
    """
    plt.figure(figsize=(8, 6))
    
    # 提取数据
    param_values = []
    accuracies = []
    
    for name, data in results.items():
        param_values.append(data[param_key])
        accuracies.append(data["best_accuracy"] * 100)
    
    # 排序
    sorted_idx = np.argsort(param_values)
    param_values = [param_values[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]
    
    # 绘制
    plt.plot(param_values, accuracies, 'o-', markersize=10, linewidth=2, color='steelblue')
    
    # 添加数值标签
    for x, y in zip(param_values, accuracies):
        plt.annotate(
            f"{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10,
        )
    
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title(f"Sensitivity Analysis: {param_name}", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()


def plot_clip_history(
    history_path: str,
    save_path: Optional[str] = None,
):
    """
    绘制裁剪阈值变化曲线
    """
    with open(history_path, "r") as f:
        history = json.load(f)
    
    plt.figure(figsize=(10, 6))
    
    rounds = history["rounds"]
    clip_values = history["clip_history"]
    
    plt.plot(rounds, clip_values, linewidth=2, color='darkorange')
    plt.fill_between(rounds, clip_values, alpha=0.3, color='orange')
    
    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel("Clipping Threshold", fontsize=12)
    plt.title("Adaptive Clipping Threshold Over Training", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()


def plot_privacy_consumption(
    history_path: str,
    epsilon_total: float,
    save_path: Optional[str] = None,
):
    """
    绘制隐私预算消耗曲线
    """
    with open(history_path, "r") as f:
        history = json.load(f)
    
    plt.figure(figsize=(10, 6))
    
    rounds = history["rounds"]
    privacy_spent = history["privacy_spent"]
    
    # 消耗曲线
    plt.plot(rounds, privacy_spent, linewidth=2, color='crimson', label='Spent')
    plt.axhline(y=epsilon_total, color='darkred', linestyle='--', linewidth=2, label=f'Budget (ε={epsilon_total})')
    
    # 填充
    plt.fill_between(rounds, privacy_spent, alpha=0.3, color='red')
    
    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel("Privacy Budget ε", fontsize=12)
    plt.title("Privacy Budget Consumption", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()


def generate_report(
    results_dir: str,
    output_dir: str = "./figures",
):
    """
    生成完整的可视化报告
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("生成实验报告...")
    
    # 查找所有实验结果
    history_files = {}
    for exp_dir in Path(results_dir).iterdir():
        if exp_dir.is_dir():
            history_path = exp_dir / "logs" / "history.json"
            if history_path.exists():
                history_files[exp_dir.name] = str(history_path)
    
    if not history_files:
        print(f"在 {results_dir} 中未找到实验结果")
        return
    
    print(f"找到 {len(history_files)} 个实验结果")
    
    # 绘制训练曲线
    if len(history_files) > 1:
        plot_training_curves(
            history_files,
            save_path=os.path.join(output_dir, "training_curves.png"),
        )
    
    # 绘制其他图表...
    print(f"\n报告已生成到: {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="实验结果可视化")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="实验结果目录")
    parser.add_argument("--output_dir", type=str, default="./figures",
                        help="图像输出目录")
    parser.add_argument("--history", type=str, default=None,
                        help="单个 history.json 文件路径")
    parser.add_argument("--plot_type", type=str, default="curves",
                        choices=["curves", "clip", "privacy", "all"],
                        help="绘制类型")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.history:
        # 单个实验可视化
        if args.plot_type in ["curves", "all"]:
            plot_training_curves(
                {"Experiment": args.history},
                save_path=os.path.join(args.output_dir, "training_curve.png"),
            )
        
        if args.plot_type in ["clip", "all"]:
            plot_clip_history(
                args.history,
                save_path=os.path.join(args.output_dir, "clip_history.png"),
            )
        
        if args.plot_type in ["privacy", "all"]:
            plot_privacy_consumption(
                args.history,
                epsilon_total=8.0,
                save_path=os.path.join(args.output_dir, "privacy_consumption.png"),
            )
    else:
        # 批量可视化
        generate_report(args.results_dir, args.output_dir)
