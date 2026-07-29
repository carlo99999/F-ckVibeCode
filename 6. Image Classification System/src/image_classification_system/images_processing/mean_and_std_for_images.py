import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def calculate_mean_and_std(
    loader: DataLoader,
    cache_path: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_path is not None and cache_path.exists():
        data = json.loads(cache_path.read_text())
        return torch.tensor(data["mean"]), torch.tensor(data["std"])

    channel_sum = torch.zeros(3)
    channel_sum_squared = torch.zeros(3)
    num_pixels = 0

    for images, _ in loader:
        batch_size, _, height, width = images.shape
        num_pixels += batch_size * height * width
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_squared += (images**2).sum(dim=[0, 2, 3])

    mean = channel_sum / num_pixels
    std = torch.sqrt((channel_sum_squared / num_pixels) - (mean**2))

    if cache_path is not None:
        cache_path.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}))

    return mean, std
