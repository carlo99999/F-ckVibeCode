from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RICE_TRAIN_DIR = DATA_DIR / "Rice_Image_Dataset_Split" / "train"
RICE_TEST_DIR = DATA_DIR / "Rice_Image_Dataset_Split" / "test"
RICE_VAL_DIR = DATA_DIR / "Rice_Image_Dataset_Split" / "val"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
CACHE_DIR = ROOT_DIR / "cache"
