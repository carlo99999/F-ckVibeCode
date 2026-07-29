from torchvision import transforms as t
from image_classification_system.images_processing.padding import PadToSquare


def transformation_for_mean_and_std() -> t.Compose:
    return t.Compose([PadToSquare(), t.Resize((224, 224)), t.ToTensor()])


def transformation_pipeline(mean, std, augment: bool = False) -> t.Compose:
    base = [PadToSquare(), t.Resize((224, 224))]
    if augment:
        base += [
            t.RandomHorizontalFlip(),
            t.RandomRotation(10),
            t.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    base += [t.ToTensor(), t.Normalize(mean, std)]
    return t.Compose(base)
