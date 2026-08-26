# Data

Place the Kaggle CSV at `data/raw/natural_disaster_human_mobility.csv`.

```bash
python scripts/download_data.py
```

Requires a Kaggle API token at `~/.kaggle/kaggle.json`.

Processed parquet files are written to `data/processed/` and are gitignored (they can be hundreds of MB).

`events.csv` is the curated disaster clock used to compute `hours_from_disaster`.
