from __future__ import annotations

"""Dataset loading and client partition helpers."""

import os
from typing import List, Literal, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DatasetName = Literal["cifar10", "mnist"]


def get_dataloader(
    num_clients: int = 500,
    batch_size: int = 32,
    alpha: float = 0.5,
    iid: bool = False,
    dataset: DatasetName = "cifar10",
    data_dir: str | None = None,
    min_samples_per_client: int = 1,
    split_max_attempts: int = 50,
) -> Tuple[List[DataLoader], DataLoader]:
    """Load dataset and split train data into client dataloaders."""

    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    if min_samples_per_client < 1:
        raise ValueError("min_samples_per_client must be >= 1")

    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "datasets")
    os.makedirs(data_dir, exist_ok=True)

    train_dataset, test_dataset = _build_dataset(dataset, data_dir)

    if iid:
        client_indices = _split_iid(
            dataset=train_dataset,
            num_clients=num_clients,
            min_samples_per_client=min_samples_per_client,
        )
    else:
        client_indices = _split_noniid_dirichlet(
            dataset=train_dataset,
            num_clients=num_clients,
            alpha=alpha,
            min_samples_per_client=min_samples_per_client,
            max_attempts=split_max_attempts,
        )

    train_loaders = [
        DataLoader(
            Subset(train_dataset, indices),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
        )
        for indices in client_indices
    ]
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=False)
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
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        return train_dataset, test_dataset

    if dataset == "mnist":
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
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        return train_dataset, test_dataset

    raise ValueError(f"Unsupported dataset: {dataset}")


def _split_iid(dataset, num_clients: int, min_samples_per_client: int) -> List[List[int]]:
    if len(dataset) < num_clients * min_samples_per_client:
        raise ValueError("dataset is too small for the requested IID split")

    indices = np.random.permutation(len(dataset)).tolist()
    client_indices = [[] for _ in range(num_clients)]

    cursor = 0
    for client_id in range(num_clients):
        next_cursor = cursor + min_samples_per_client
        client_indices[client_id].extend(indices[cursor:next_cursor])
        cursor = next_cursor

    remaining = indices[cursor:]
    for offset, index in enumerate(remaining):
        client_indices[offset % num_clients].append(index)

    for bucket in client_indices:
        np.random.shuffle(bucket)
    return client_indices


def _split_noniid_dirichlet(
    dataset,
    num_clients: int,
    alpha: float,
    min_samples_per_client: int = 1,
    max_attempts: int = 50,
) -> List[List[int]]:
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if len(dataset) < num_clients * min_samples_per_client:
        raise ValueError("dataset is too small for the requested min_samples_per_client")

    targets = _extract_targets(dataset)
    num_classes = len(np.unique(targets))

    for _ in range(max_attempts):
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]

        for class_id in range(num_classes):
            class_indices = np.where(targets == class_id)[0]
            np.random.shuffle(class_indices)

            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            splits = np.random.multinomial(len(class_indices), proportions)

            offset = 0
            for client_id, count in enumerate(splits):
                if count <= 0:
                    continue
                client_indices[client_id].extend(class_indices[offset : offset + count].tolist())
                offset += count

        if _ensure_minimum_samples(client_indices, min_samples_per_client):
            for indices in client_indices:
                np.random.shuffle(indices)
            return client_indices

    raise RuntimeError(
        "Failed to produce a non-IID split satisfying min_samples_per_client; "
        "try reducing min_samples_per_client or increasing split_max_attempts."
    )


def _ensure_minimum_samples(client_indices: List[List[int]], min_samples_per_client: int) -> bool:
    """Repair small-client tails by moving surplus samples from large clients."""

    deficits = [idx for idx, samples in enumerate(client_indices) if len(samples) < min_samples_per_client]
    if not deficits:
        return True

    donors = [idx for idx, samples in enumerate(client_indices) if len(samples) > min_samples_per_client]
    donor_ptr = 0

    for client_id in deficits:
        needed = min_samples_per_client - len(client_indices[client_id])
        while needed > 0:
            while donor_ptr < len(donors) and len(client_indices[donors[donor_ptr]]) <= min_samples_per_client:
                donor_ptr += 1
            if donor_ptr >= len(donors):
                return False

            donor_id = donors[donor_ptr]
            moved_index = client_indices[donor_id].pop()
            client_indices[client_id].append(moved_index)
            needed -= 1

    return min(len(samples) for samples in client_indices) >= min_samples_per_client


def _extract_targets(dataset) -> np.ndarray:
    targets = dataset.targets
    if isinstance(targets, list):
        return np.array(targets)
    return targets.numpy() if hasattr(targets, "numpy") else np.array(targets)
