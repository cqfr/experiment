from __future__ import annotations

"""Dataset loading and client partition helpers."""

import os
from typing import List, Literal, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DatasetName = Literal["cifar10", "mnist"]


def get_dataloader(
    num_clients: int = 10,
    batch_size: int = 32,
    alpha: float = 0.5,
    iid: bool = False,
    dataset: DatasetName = "cifar10",
    data_dir: str | None = None,
) -> Tuple[List[DataLoader], DataLoader]:
    """Load dataset and split train data into client dataloaders."""

    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")

    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "datasets")
    os.makedirs(data_dir, exist_ok=True)

    train_dataset, test_dataset = _build_dataset(dataset, data_dir)

    if iid:
        client_indices = _split_iid(train_dataset, num_clients)
    else:
        client_indices = _split_noniid_dirichlet(train_dataset, num_clients, alpha)

    train_loaders = [
        DataLoader(Subset(train_dataset, indices), batch_size=batch_size, shuffle=True)
        for indices in client_indices
    ]
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return train_loaders, test_loader


def _build_dataset(dataset: DatasetName, data_dir: str):
    if dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        return train_dataset, test_dataset

    if dataset == "mnist":
        # Convert to 3x32x32 for a shared ResNet-18 pipeline.
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.1307, 0.1307, 0.1307),
                    std=(0.3081, 0.3081, 0.3081),
                ),
            ]
        )
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        return train_dataset, test_dataset

    raise ValueError(f"Unsupported dataset: {dataset}")


def _split_iid(dataset, num_clients: int) -> List[List[int]]:
    indices = np.random.permutation(len(dataset))
    return [indices[i::num_clients].tolist() for i in range(num_clients)]


def _split_noniid_dirichlet(dataset, num_clients: int, alpha: float) -> List[List[int]]:
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    targets = _extract_targets(dataset)
    num_classes = len(np.unique(targets))
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(targets == cls)[0]
        np.random.shuffle(cls_idx)

        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        splits = (proportions * len(cls_idx)).astype(int)
        splits[-1] = len(cls_idx) - splits[:-1].sum()

        start = 0
        for client_id, count in enumerate(splits):
            client_indices[client_id].extend(cls_idx[start : start + count].tolist())
            start += count

    for indices in client_indices:
        np.random.shuffle(indices)

    return client_indices


def _extract_targets(dataset) -> np.ndarray:
    targets = dataset.targets
    if isinstance(targets, list):
        return np.array(targets)
    return targets.numpy() if hasattr(targets, "numpy") else np.array(targets)


if __name__ == "__main__":
    loaders, test_loader = get_dataloader(num_clients=5, dataset="mnist")
    print(len(loaders), len(test_loader.dataset))
