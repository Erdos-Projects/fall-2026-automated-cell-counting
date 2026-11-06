# Plan

> **Overview** · [All notes](README.md)

One row per weekly check-in. Scope, data and KPIs are in [`project.md`](project.md).

| Week | Check-in | What we deliver | Status |
|---|---|---|---|
| 1 | Scope, data sources, stakeholders, KPIs | [`project.md`](project.md) | ✅ Done |
| 2 | EDA and preprocessing | [`01_eda_overview.ipynb`](../notebooks/01_eda_overview.ipynb), the shared split [`splits.csv`](../splits/splits.csv), [`00_data.md`](00_data.md) | ✅ Done |
| 3 | Model architecture: baseline and deep learning | Baseline B0 (session mean, reference) and B1/B2 Cellpose-SAM (default / tuned on train; B2 is the baseline), then our own model: YOLO26n trained on our boxes ([`02_baseline_explained.md`](02_baseline_explained.md)) | ✅ Done |
| 4 | Baseline performance | [`02_baseline_cellpose.ipynb`](../notebooks/02_baseline_cellpose.ipynb), [`02_baseline_results.md`](02_baseline_results.md) | ✅ Done |
| 5 | Deep-learning model, iteration 1 | Our model, YOLO26n, trained with default settings and scored like the baseline: MAE 1.6 vs 5.6 cells per image ([`03_yolo_training.md`](03_yolo_training.md), [`04_yolo_performance.md`](04_yolo_performance.md)) | ✅ Done |
| 6 | Deep-learning model, iteration 2 | YOLO26n tested one change at a time: augmentation screen and RGB vs grayscale ([`03_yolo_training.md`](03_yolo_training.md), [`03a_rgb_grayscale_results.md`](03a_rgb_grayscale_results.md)), and applied to other datasets ([`05_extra_experiments.md`](05_extra_experiments.md)) | ✅ Done |
| 7 | Final project | Clean repo, 1-page executive summary, 5-minute presentation | ✅ Done |

## How we evaluate

- **Data:** the 500 Roboflow images, split by [`splits/splits.csv`](../splits/splits.csv) (351 train / 74 valid / 75 test).
- **Task for now:** the total number of cells per image. Telling Live from Dead comes later.
- **Scores:** total-count MAE (cells per image), ±10% accuracy, R², and seconds per image. For models that find cells, we also report precision and recall.
- **Fair settings:** anything we tune is chosen on train (or valid), never on test.
- **Two test levels:** the random split for every model, and leave-one-session-out (train on 4 sessions, test on the 5th) for the baselines. The session alone predicts the count with R² ≈ 0.7, so a model has to beat that.
