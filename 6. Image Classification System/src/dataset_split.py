"""
Creates train/test/val splits for the Rice Image Dataset using symlinks.
Split ratios: 80% train, 15% test, 5% val.
"""

import random
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.80
TEST_RATIO = 0.15
# val = remaining 0.05

ROOT_DIR = Path(__file__).parent.parent
DATASET_DIR = ROOT_DIR / "data" / "Rice_Image_Dataset"
SPLITS_DIR = ROOT_DIR / "data" / "Rice_Image_Dataset_Split"

SPLITS = ("train", "test", "val")


def split_class(class_dir: Path) -> None:
    images = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))

    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_test = int(n * TEST_RATIO)

    splits_images = {
        "train": images[:n_train],
        "test": images[n_train : n_train + n_test],
        "val": images[n_train + n_test :],
    }

    for split, split_images in splits_images.items():
        dest_dir = SPLITS_DIR / split / class_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for img_path in split_images:
            link = dest_dir / img_path.name
            if not link.exists():
                link.symlink_to(img_path.resolve())

    counts = {k: len(v) for k, v in splits_images.items()}
    print(f"  {class_dir.name}: train={counts['train']}, test={counts['test']}, val={counts['val']}")


def main() -> None:
    random.seed(SEED)

    class_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found in {DATASET_DIR}")

    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {SPLITS_DIR}")
    print(f"Classes: {[d.name for d in class_dirs]}\n")

    for class_dir in sorted(class_dirs):
        split_class(class_dir)

    print("\nDone.")
    for split in SPLITS:
        total = sum(1 for _ in (SPLITS_DIR / split).rglob("*") if _.is_file() or _.is_symlink())
        print(f"  {split}: {total} images")


if __name__ == "__main__":
    main()
