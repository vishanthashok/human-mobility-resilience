# Human Mobility Resilience During Natural Disasters

Analyzed geotagged tweets across 15 natural disasters to quantify **individual mobility disruption** and model **time-to-recovery** with geospatial features and survival analysis.

The Kaggle dataset does not include recovery time. This project builds that target from each person’s own baseline movement.

## Core question

How much does a disaster disrupt a person’s normal movement pattern, and how long does it take them to return toward baseline behavior?

## Data

[Human Mobility During Natural Disasters](https://www.kaggle.com/datasets/dryad/human-mobility-during-natural-disasters) (Wang & Taylor 2016, PLoS ONE).

| Column | Meaning |
|---|---|
| `disaster.event` | One of 15 events (typhoons, earthquakes, winter storms, thunderstorms, wildfires) |
| `user.anon` | Anonymous user id, unique **within** each event |
| `latitude` | Tweet latitude |
| `longitude.anon` | Longitude shifted for anonymity |
| `time` | Tweet timestamp |

Longitude is anonymized with a shift. Distances and radius of gyration remain valid **within** an event. Do not treat coordinates as true map locations.

## Method (short)

1. Attach a curated disaster clock (`data/events.csv`) and compute `hours_from_disaster` / `day_relative`.
2. Keep users with enough data: 10 observations in the baseline window (days −14 to −2), 5 during the 72-hour impact window, 10 after.
3. Estimate home as the median baseline coordinate. Build daily metrics: radius of gyration, distance traveled, displacement from home, unique locations.
4. **Mobility Disruption Score** = 0.4·radius deviation + 0.3·displacement deviation + 0.3·daily-distance deviation.
5. **Recovered** when all four metrics stay within 20% of that user’s baseline for two consecutive days. Users who never recover in the observation window are right-censored.
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

## Run

Napa earthquake first (debug the pipeline on one sudden-onset event):

```bash
python scripts/run_pipeline.py --events 13_Napa
```

Three-disaster contrast (earthquake vs winter storm vs typhoon):

```bash
python scripts/run_pipeline.py --events 13_Napa,23_Atlanta,04_Rammasun_Manila
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
- later: train on 12 disasters, validate on 2, test on 1 unseen disaster.

## License and attribution

Research data: Wang Q, Taylor JE (2016). *Patterns and limitations of urban human mobility resilience under the influence of multiple types of natural disaster.* PLoS ONE 11(1): e0147299. Dryad: https://doi.org/10.5061/dryad.88354
