# HousingMarketClassifier

A machine learning classifier that predicts Norwegian housing market risk (Hot / Stable / Cooling) one quarter ahead, across 15 counties, using macroeconomic data from Statistics Norway (SSB).

**Headline finding: the model does not beat a seasonal baseline.** See [Results](#results). Earlier versions of this README reported 86.5% test accuracy; that number came from a leaking train/test split and has been retracted.

## Approach

I started by building a **Random Forest classifier from scratch** — decision tree construction with Gini impurity, bootstrap sampling, feature subsampling, and majority voting — to understand the fundamentals. The initial model trained on a small manually downloaded dataset (80 samples, 5 regions, 2020–2023) achieved 67% validation accuracy and couldn't predict the "Stable" class at all.

To improve, I built an **automated SSB API data pipeline** (`fetch_ssb_data.py`) that fetches 10 statistical tables programmatically, covering 2005–2024 across all 15 Norwegian counties. A large part of the work sits in `data_parser.py`, which harmonises the 2020 and 2024 **county mergers and splits** (Viken → Østfold/Akershus/Buskerud, Vestfold og Telemark → Vestfold/Telemark, Troms og Finnmark → Troms/Finnmark) onto one consistent set of 15 modern counties so that a series is comparable across the whole period.

With the larger dataset, I pivoted to **scikit-learn's ensemble classifiers**, comparing `RandomForestClassifier` (with balanced class weights) against `GradientBoostingClassifier`.

[Claude Opus 4.6](https://www.anthropic.com/claude) was used as a development tool throughout the project.

## The leakage bug, and the fix

The evaluation was wrong, and fixing it changed the conclusions of the project.

`create_labels` returned its frame sorted by `['region', 'quarter']`. `train_model.py` then took the first 65% of **rows** as training data under a comment that said `# Time-based split`. Because the rows were sorted by county name first, this split the panel **alphabetically by county**, not by time:

| | Old (broken) split | New split |
|---|---|---|
| Train | Agder → Telemark, all years | 2014K1–2020K3 |
| Validation | Troms, Trøndelag, most of Vestfold | 2020K4–2022K4 |
| Test | rest of Vestfold, Vestland, Østfold | 2023K1–2024K3 |

That leaked in two ways:

1. **Shared labels.** The 15 counties map to only **10 distinct SSB price regions**. Buskerud, Telemark, Vestfold and Østfold share one price series, so their labels are *identical* every quarter. Most of the old test set had the same answer as a Buskerud or Telemark row the model had trained on for that same quarter.
2. **Quarters in both sets.** All 79 quarters appeared in train and test simultaneously. National series (CPI, policy rate, GDP) are unique per quarter, so together they act as a timestamp — the model could learn "2016K2 was Hot" from one county and apply it to another.

The fix assigns **whole quarters** to each set, so every county's rows for a quarter land together and the model is only ever scored on quarters it has never seen.

Two related bugs were fixed at the same time:

- `rate_change`, `unemployment_change`, `mortgage_rate_change` and `gdp_ma4` used `.diff()` / `.rolling()` **without grouping by region**, so on region-sorted data the first row of each county was computed against the previous county's last row. These are now grouped.
- `supply_demand_score` and `demand_indicator` were z-scored against statistics of the **whole panel**, including the test years. They now use per-region expanding (past-only) statistics.

## Results

Evaluated on **unseen quarters** (2023K1–2024K3), 645 county-quarters from 2014 onward, 32 features.

| Model / baseline | Val macro-F1 | Test macro-F1 | Test accuracy |
|---|---|---|---|
| Majority class | 0.221 | 0.197 | 41.9% |
| Persistence (repeat last quarter's label) | 0.322 | 0.357 | 44.8% |
| **Seasonal lookup (quarter-of-year only)** | **0.627** | **0.642** | **92.4%** |
| Random Forest (32 features) | 0.590 | 0.639 | 91.4% |
| Gradient Boosting (32 features) | 0.508 | — | — |

**The Random Forest does not beat the seasonal baseline** (−0.003 macro-F1 on test). A lookup table containing nothing but "which quarter of the year is it" matches the full 32-feature model.

This is the honest result, and it is the interesting one. Norwegian house prices have a strong within-year cycle — in the training window 83% of Q1s are Hot and 77% of Q3s are Cooling — and because the label is next quarter's price change bucketed into three bins, the calendar alone almost determines the answer. An ablation confirms it: removing `seasonal_factor` and `quarter_num` drops the model to 0.555 test macro-F1, so nearly all of its apparent skill is seasonality, and the macro series add nothing on top of a calendar.

Per-class test performance (Random Forest):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Hot | 0.98 | 1.00 | 0.99 | 44 |
| Stable | 0.00 | 0.00 | 0.00 | 8 |
| Cooling | 0.88 | 0.98 | 0.93 | 53 |

The 91.4% accuracy is not the headline it looks like: the model never predicts Stable at all, and Stable is only 8 of 105 test rows. Macro-F1 is the metric to read here.

### Confusion Matrix

![Confusion Matrix](output/confusion_matrix.png)

### Feature Importance

![Feature Importance](output/feature_importance.png)

The top two features are `seasonal_factor` and `quarter_num` — which is the finding above, visible directly in the importances.

### Where this would go next

The task as defined is close to a seasonality lookup, so the model has little room to show skill. The useful next step is to make the target harder and more meaningful: label on **seasonally-adjusted** or year-over-year price changes rather than raw quarter-on-quarter, so the cycle is removed from the target and the macro features have to carry real signal. Redefining the label changes what the project claims to predict, so it is left as explicit future work rather than folded in quietly.

## Known limitations

- **15 counties, 10 price series.** SSB's price index (table 07221) covers 10 regions. Buskerud/Telemark/Vestfold/Østfold share one series and Finnmark/Nordland/Troms share another, so those counties have identical labels. The effective sample is smaller than 645 rows suggests.
- **2014 onward only.** The policy rate and mortgage rate series start in 2014. The previous code ran on 2005–2024 and passed missing values through `fillna(0)`, which told the model that Norway had a 0% policy rate for a decade — indistinguishable from 2020, when the rate genuinely was 0%. The run is now restricted to 2014+ with median imputation fit on the training window only. `--min-year 2005` runs the longer panel; features with under 70% coverage in the training window are dropped automatically rather than zero-filled.
- **Household income dropped.** SSB table 06944 has no data for 7 of the 15 counties, so `household_income` and the affordability features derived from it are dropped by the coverage rule.
- **Small test set.** 7 quarters, 105 rows, only 8 of them Stable. Treat per-class figures for Stable as indicative at best.
- **`backtest_strategy.py` was removed.** It referenced undefined variables and never ran, and the leveraged "profit" and Sharpe figures it computed from a price index were not meaningful.

## Data Sources

All data fetched automatically from the [SSB Statistikkbanken API](https://www.ssb.no/api/):

| SSB Table | Feature | Frequency | Coverage |
|-----------|---------|-----------|----------|
| 03013 | Consumer Price Index (CPI) | Monthly | 2005–2025 |
| 10701 | Norges Bank policy rate | Monthly | 2014–2026 |
| 01222 | Population change by county | Quarterly | 2005–2024 |
| 07221 | House price index by region | Quarterly | 2005–2024 |
| 10187 | Property sales volume | Quarterly | 2008–2024 |
| 13760 | Unemployment rate | Monthly | 2006–2026 |
| 03723 | Building starts by county | Monthly | 2005–2026 |
| 10748 | Mortgage interest rates | Monthly | 2014–2025 |
| 09171 | GDP volume change | Quarterly | 2005–2025 |
| 06944 | Household income by county | Annual | 2005–2023 |

## Labels

Each county-quarter is labelled by the **next** quarter's change in the price index (features use only current and past data):

| Label | Condition |
|-------|-----------|
| Hot | next-quarter price change > +2.0% |
| Stable | between −0.5% and +2.0% |
| Cooling | next-quarter price change < −0.5% |

## Pipeline

```
fetch_ssb_data.py  →  data_parser.py  →  enhanced_features.py  →  train_model.py
     (API)             (JSON→CSV)         (35 features)        (split, baselines, sklearn)
```

```bash
# Full pipeline (fetch + parse + train)
python run_complete_analysis.py

# Skip API fetch (use cached JSON)
python run_complete_analysis.py --skip-fetch
```

```bash
# Train on the longer 2005+ panel instead (drops the rate features)
python src/train_model.py --min-year 2005
```

## Project Structure

```
├── run_complete_analysis.py       # Pipeline orchestrator (entry point)
├── src/
│   ├── fetch_ssb_data.py          # SSB API data fetcher (10 tables)
│   ├── data_parser.py             # JSON-stat2 parser + county-merger harmonisation
│   ├── enhanced_features.py       # Feature engineering & labeling
│   ├── train_model.py             # Chronological split, baselines, training
│   └── random_forest_classifier.py  # Original from-scratch RF (reference)
├── data/                          # Cached SSB JSON data (10 JSON files)
├── output/                        # Generated artifacts
│   ├── processed_data.csv
│   ├── results.json               # Metrics, written by train_model.py
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── requirements.txt
```

`output/results.json` is written on every run so the numbers in this README can be checked against the code rather than drifting from it.

## Requirements

```bash
pip install -r requirements.txt
```
