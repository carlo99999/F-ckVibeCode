from pathlib import Path

import torch
from torchvision import datasets, transforms as t
from torch.utils.data import DataLoader, WeightedRandomSampler


def create_dataloader(
    path: str | Path,
    transformations: t.Compose,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = False,
    balanced: bool = False,
) -> DataLoader:
    dataset = datasets.ImageFolder(path, transform=transformations)

    sampler = None
    if balanced:
        class_counts = torch.zeros(len(dataset.classes))
        for _, label in dataset.samples:
            class_counts[label] += 1
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label].item() for _, label in dataset.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
