"""
data/utils.py
CIFAR-10 数据加载 + IID/Non-IID 划分
返回各客户端的 DataLoader 列表
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_dataloader(
    num_clients: int = 10,
    batch_size: int = 32,
    alpha: float = 0.5,
    iid: bool = False,
    data_dir: str = None,  # ✅ 修复: 默认 None,自动推断
) -> tuple[list[DataLoader], DataLoader]:
    """
    加载 CIFAR-10 并按客户端划分数据，返回 DataLoader 列表。
 
    参数：
        num_clients : 客户端总数
        batch_size  : 每个客户端训练时的 batch 大小
        alpha       : Dirichlet 分布参数，越小数据越异构（Non-IID 时使用）
        iid         : True = IID 均匀划分；False = Non-IID Dirichlet 划分
        data_dir    : 数据集下载/缓存路径 (None = 自动使用相对路径)
 
    返回：
        train_loaders : list[DataLoader]，长度为 num_clients
        test_loader   : DataLoader，全局测试集（所有客户端共用）
    """
    
    # ✅ 修复: 自动推断数据路径
    if data_dir is None:
        # 推荐用法: 在项目根目录下创建 data/datasets
        data_dir = os.path.join(os.path.dirname(__file__), "datasets")
    
    # 确保目录存在
    os.makedirs(data_dir, exist_ok=True)
 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
 
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
 
    # 按划分策略拿到每个客户端的样本索引列表
    if iid:
        client_indices = _split_iid(train_dataset, num_clients)
    else:
        client_indices = _split_noniid_dirichlet(train_dataset, num_clients, alpha)
 
    # 为每个客户端构建 DataLoader
    train_loaders = [
        DataLoader(
            Subset(train_dataset, indices),
            batch_size=batch_size,
            shuffle=True,
        )
        for indices in client_indices
    ]
 
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
 
    return train_loaders, test_loader


# ──────────────────────────────────────────────
# 内部工具函数
# ──────────────────────────────────────────────

def _split_iid(dataset, num_clients: int) -> list[list[int]]:
    """均匀随机划分：每个客户端样本数相等，类别分布与全局一致。"""
    n = len(dataset)
    indices = np.random.permutation(n)
    return [indices[i::num_clients].tolist() for i in range(num_clients)]


def _split_noniid_dirichlet(
    dataset, num_clients: int, alpha: float
) -> list[list[int]]:
    """
    Dirichlet Non-IID 划分：
    对每个类别，用 Dirichlet(alpha) 采样各客户端的分配比例。
    alpha 越小，数据分布越不均匀（极端情况下每个客户端只有1-2个类）。
    """
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    client_indices = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(targets == cls)[0]
        np.random.shuffle(cls_idx)

        # 用 Dirichlet 分布决定这个类的样本如何分配给各客户端
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        # 按比例切分
        splits = (proportions * len(cls_idx)).astype(int)
        # 修正舍入误差，确保总数不变
        splits[-1] = len(cls_idx) - splits[:-1].sum()

        start = 0
        for client_id, count in enumerate(splits):
            client_indices[client_id].extend(cls_idx[start:start + count].tolist())
            start += count

    # 打乱每个客户端内部顺序
    for indices in client_indices:
        np.random.shuffle(indices)

    return client_indices


# ──────────────────────────────────────────────
# 快速验证（直接运行本文件时执行）
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== IID 划分 ===")
    train_loaders, test_loader = get_dataloader(num_clients=100, iid=True)
    for i, loader in enumerate(train_loaders):
        print(f"  客户端 {i:02d}: {len(loader.dataset)} 条样本")
    print(f"  测试集: {len(test_loader.dataset)} 条样本")

    print("\n=== Non-IID 划分 (α=0.5) ===")
    train_loaders, test_loader = get_dataloader(num_clients=100, iid=False, alpha=0.1)
    for i, loader in enumerate(train_loaders):
        targets = np.array(loader.dataset.dataset.targets)[loader.dataset.indices]
        unique, counts = np.unique(targets, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"  客户端 {i:02d}: {len(targets):4d} 条 | 类别分布: {dist}")