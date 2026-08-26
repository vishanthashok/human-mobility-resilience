"""Project paths and analysis constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
EVENTS_PATH = DATA_DIR / "events.csv"
RAW_CSV_NAME = "natural_disaster_human_mobility.csv"

# Wang & Taylor 2016 used a 72-hour perturbation window.
BASELINE_DAY_START = -14
BASELINE_DAY_END = -2  # inclusive; day -1 excluded (warnings / evacuation)
DURING_DAYS = (0, 1, 2)
MIN_OBS_BASELINE = 10
MIN_OBS_DURING = 5
MIN_OBS_AFTER = 10

# Spatial cell size: 3 decimal degrees ≈ 100 m.
LOCATION_DECIMALS = 3

# Earth radius used in Wang & Taylor 2016 (meters → km).
EARTH_RADIUS_KM = 6367.0

# Relative-deviation floors so near-zero baselines do not explode.
EPS_KM = 0.1
EPS_COUNT = 1.0

MDS_WEIGHTS = {
    "radius": 0.4,
    "displacement": 0.3,
    "mobility_distance": 0.3,
}

RECOVERY_TOLERANCE = 0.20
RECOVERY_CONSECUTIVE_DAYS = 2

# IDs match the Kaggle CSV (they differ from the Kaggle dataset card).
FOCUS_EVENT = "14_Napa"
V15_EVENTS = ("14_Napa", "23_Atlanta", "08_Rammasun_Manila")

RANDOM_STATE = 42
N_CV_FOLDS = 5
