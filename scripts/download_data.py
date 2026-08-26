"""Download the Dryad/Kaggle human-mobility CSV into data/raw/."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RAW_CSV_NAME, RAW_DIR  # noqa: E402


def main() -> None:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading dryad/human-mobility-during-natural-disasters …")
    cache_path = Path(kagglehub.dataset_download("dryad/human-mobility-during-natural-disasters"))
    csv_files = list(cache_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found under {cache_path}")
    preferred = [p for p in csv_files if p.name == RAW_CSV_NAME]
    src = preferred[0] if preferred else csv_files[0]
    dest = RAW_DIR / RAW_CSV_NAME
    shutil.copy2(src, dest)
    print(f"Wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
