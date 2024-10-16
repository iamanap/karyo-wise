from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")


DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_AUTOKARY_DIR = RAW_DATA_DIR / "AutoKary2022_2200x3200"
RAW_DATA_DIR_BBOX_DATA = RAW_DATA_DIR / "bbox_data"
RAW_DATA_DIR_BBOX_DATA_IMG = RAW_DATA_DIR_BBOX_DATA / "JEPG"
RAW_DATA_DIR_BBOX_DATA_ANN = RAW_DATA_DIR_BBOX_DATA / "annotations"

RAW_DATA_DIR_SEG_DATA = RAW_DATA_DIR / "seg_data"
RAW_DATA_DIR_SEG_DATA_TRAIN = RAW_DATA_DIR_SEG_DATA / "train"
RAW_DATA_DIR_SEG_DATA_TEST = RAW_DATA_DIR_SEG_DATA / "test"

INTERIM_DATA_DIR = DATA_DIR / "interim"
INTERIM_DATA_DIR_TRAIN_IMG = INTERIM_DATA_DIR / "train_img"
INTERIM_DATA_DIR_TRAIN_ANN = INTERIM_DATA_DIR / "train_ann"
INTERIM_DATA_DIR_TEST_IMG = INTERIM_DATA_DIR / "test_img"
INTERIM_DATA_DIR_TEST_ANN = INTERIM_DATA_DIR / "test_ann"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR_TRAIN_IMG = PROCESSED_DATA_DIR / "train_img"
PROCESSED_DATA_DIR_TRAIN_MASKS = PROCESSED_DATA_DIR / "train_masks"
PROCESSED_DATA_DIR_TEST_IMG = PROCESSED_DATA_DIR / "test_img"
PROCESSED_DATA_DIR_TEST_MASKS = PROCESSED_DATA_DIR / "test_masks"
TRAIN_DATA_CSV = PROCESSED_DATA_DIR / "seg_train_data.csv"
TEST_DATA_CSV = PROCESSED_DATA_DIR / "seg_test_data.csv"

EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SAM_WEIGHTS = EXTERNAL_DATA_DIR / "sam_vit_h_4b8939.pth"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
