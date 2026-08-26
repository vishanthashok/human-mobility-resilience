# Human Mobility Resilience During Natural Disasters

Analyzed **4.69 million** geolocated tweets across **15 natural disasters** to quantify individual mobility disruption and model time-to-recovery with geospatial features and survival analysis.

The Kaggle dataset does not include recovery time. This project builds that target from each person’s own baseline movement.

## Core question

How much does a disaster disrupt a person’s normal movement pattern, and how long does it take them to return toward baseline behavior?

## Headline results

After requiring 10 baseline, 5 during, and 10 post-impact observations per person:

| | |
|---|---|
| Eligible users | 7,402 (14 of 15 events; Detroit has no pre-disaster window in this extract) |
| Recovered inside the observation window | 1,386 |
| Right-censored | 6,016 |
| Median recovery (uncensored) | 3 days |
| Random forest C-index (user GroupKFold) | 0.83 |
| Unseen-disaster holdout (Bohol earthquake) | C-index 0.85, RMSE 3.8 days |

Kaplan–Meier fraction recovered: **10% by day 3**, 14% by day 7, 19% by day 14, 21% by day 21. Most people are censored because several event extracts have short follow-up (Napa ends at day +5) and recovery requires two consecutive days inside a 20% band of that person’s baseline.

Research associations (Spearman vs days-to-recovery, uncensored users):

- Larger **day-0 disruption** predicts longer recovery (ρ = 0.47).
- Larger **initial displacement from home** predicts longer recovery (ρ = 0.40).
- People with a **larger baseline radius of gyration do not recover faster** (ρ = 0.26 with recovery days — they take longer).

Features used for prediction are leakage-safe: baseline behavior plus the first 24 hours after impact only.

## Data

[Human Mobility During Natural Disasters](https://www.kaggle.com/datasets/dryad/human-mobility-during-natural-disasters) (Wang & Taylor 2016, PLoS ONE).

| Column | Meaning |
|---|---|
| `disaster.event` | One of 15 events (typhoons, earthquakes, winter storms, thunderstorms, wildfires) |
| `user.anon` | Anonymous user id, unique **within** each event |
| `latitude` | In this extract: geographic **longitude** |
| `longitude.anon` | Shifted **latitude** (anonymized) |
| `time` | Tweet timestamp (treated as UTC) |

Ingest swaps the coordinate columns. Distances are valid within an event; do not plot on a real basemap. CSV event ids also differ from the Kaggle card: `14_Napa`, `08_Rammasun_Manila`, `06_Kalmaegi`, `12_Bohol`, `13_Iquique`.

## Method (short)

1. Attach a curated disaster clock (`data/events.csv`) and compute `hours_from_disaster` / `day_relative`.
2. Keep users with enough data: 10 observations in the baseline window (days −14 to −2), 5 during the 72-hour impact window, 10 after.
3. Estimate home as the median baseline coordinate. Build daily metrics: radius of gyration, distance traveled, displacement from home, unique locations.
4. **Mobility Disruption Score** = 0.4·radius deviation + 0.3·displacement deviation + 0.3·daily-distance deviation.
5. **Recovered** when radius, displacement, and daily distance all stay within 20% of that user’s baseline for two consecutive days. Users who never recover in the observation window are right-censored.
6. Predict `days_to_recovery` using **only baseline + first 24 hours** (no leakage from later days). Survival models (Kaplan–Meier, Cox, Random Survival Forest) handle censoring.

## Project layout

```
data/events.csv          curated disaster timestamps
notebooks/               narrative analysis
src/                     reusable pipeline
dashboard/app.py         Streamlit resilience dashboard
scripts/                 download + run
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
```

Kaggle credentials are required for `kagglehub` (`~/.kaggle/kaggle.json`).

On macOS, XGBoost needs OpenMP (`brew install libomp`). If it is missing, the pipeline still runs linear regression and random forest.

## Run

Napa earthquake first (debug the pipeline on one sudden-onset event):

```bash
python scripts/run_pipeline.py --events 14_Napa
```

Three-disaster contrast (earthquake vs winter storm vs typhoon):

```bash
python scripts/run_pipeline.py --events 14_Napa,23_Atlanta,08_Rammasun_Manila
```

All 15 events:

```bash
python scripts/run_pipeline.py --events all
```

Dashboard:

```bash
streamlit run dashboard/app.py
```

Notebooks: `notebooks/01_eda.ipynb` through `05_modeling.ipynb`.

## Leakage and splits

If the prediction is made 24 hours after impact, features may use baseline behavior, disaster type, and day-0 post-impact metrics only.

- v1 (single disaster): split **by user**, never by tweet row.
- Full run: user-level GroupKFold, plus a holdout disaster never seen in training (Bohol in the current artifacts).