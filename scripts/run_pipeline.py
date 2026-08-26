"""Run the mobility-resilience pipeline for one, three, or all disasters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FOCUS_EVENT  # noqa: E402
from src.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        default=FOCUS_EVENT,
        help="Comma-separated event ids, or 'all'. Default: 14_Napa",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Stop after feature engineering and recovery labels.",
    )
    args = parser.parse_args()
    bundle = run(event_spec=args.events, skip_models=args.skip_models)
    users = bundle["tables"]["users"]
    print(
        f"Users: {len(users)} | recovered: {int(users['event_observed'].sum())} | "
        f"censored: {int((users['event_observed'] == 0).sum())}"
    )
    print(f"Disasters: {sorted(users['disaster'].unique())}")
    if bundle["models"]:
        print(bundle["models"]["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
