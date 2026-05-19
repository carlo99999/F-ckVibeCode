import torch
from torch.utils.data import DataLoader


def calculate_mean_and_std(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    channel_sum = torch.zeros(3)
    channel_sum_squared = torch.zeros(3)
    num_pixels = 0

    for images, _ in loader:
        batch_size, channels, height, width = images.shape
        num_pixels += batch_size * height * width
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_squared += (images**2).sum(dim=[0, 2, 3])

    mean = channel_sum / num_pixels
    std = torch.sqrt((channel_sum_squared / num_pixels) - (mean**2))

    return mean, std
