from pathlib import Path

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
) -> tuple[DataLoader, list[float], list[float]]:
    transformation_mean_std = transformation_for_mean_and_std()

    mean_std_dataloader = create_dataloader(
        path, transformations=transformation_mean_std, batch_size=32
    )

    mean, std = calculate_mean_and_std(mean_std_dataloader)

    full_transformation_pipeline = transformation_pipeline(mean=mean, std=std)

    train_dataloader = create_dataloader(
        path, transformations=full_transformation_pipeline, batch_size=batch_size
    )

    return train_dataloader, mean, std


def create_test_dataloader_pipeline(
    path: str | Path,
    mean,
    std,
    batch_size: int = 128,
) -> DataLoader:
    full_transformation_pipeline = transformation_pipeline(mean=mean, std=std)

    return create_dataloader(
        path, transformations=full_transformation_pipeline, batch_size=batch_size
    )
