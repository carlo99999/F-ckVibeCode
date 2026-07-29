from pathlib import Path

from torch import Tensor
from torch.utils.data import DataLoader

from image_classification_system.images_processing.dataloaders import create_dataloader
from image_classification_system.images_processing.mean_and_std_for_images import (
    calculate_mean_and_std,
)
from image_classification_system.images_processing.transformations import (
    transformation_for_mean_and_std,
    transformation_pipeline,
)


def create_train_dataloader_pipeline(
    path: str | Path,
    batch_size: int = 128,
    cache_dir: Path | None = None,
) -> tuple[DataLoader, Tensor, Tensor]:
    mean_std_dataloader = create_dataloader(
        path, transformations=transformation_for_mean_and_std(), batch_size=32
    )

    cache_path = (cache_dir / "mean_std.json") if cache_dir is not None else None
    mean, std = calculate_mean_and_std(mean_std_dataloader, cache_path=cache_path)

    train_dataloader = create_dataloader(
        path,
        transformations=transformation_pipeline(mean=mean, std=std, augment=True),
        batch_size=batch_size,
        balanced=True,
    )

    return train_dataloader, mean, std


def create_test_dataloader_pipeline(
    path: str | Path,
    mean,
    std,
    batch_size: int = 128,
) -> DataLoader:
    return create_dataloader(
        path,
        transformations=transformation_pipeline(mean=mean, std=std),
        batch_size=batch_size,
    )
