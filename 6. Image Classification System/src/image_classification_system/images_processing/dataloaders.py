from pathlib import Path

from torchvision import datasets, transforms as t
from torch.utils.data import DataLoader

def create_dataloader(path: str | Path, transformations: t.Compose, batch_size: int = 64) -> DataLoader:
    dataset = datasets.ImageFolder(path, transform=transformations)
    return DataLoader(dataset=dataset, batch_size=batch_size)