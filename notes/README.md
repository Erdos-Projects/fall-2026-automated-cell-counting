# Notes

The team's write-ups, in the order of the project's story. Each note starts with the number of the notebooks it documents, so `02_baseline_results.md` goes with `notebooks/02_baseline_cellpose.ipynb`. Each note also opens with a header line that gives its notebooks and links back here.

## Overview

- [Project](project.md): scope, data sources, stakeholders, KPIs
- [Plan](plan.md): weekly roadmap, one row per check-in

## Contents

| # | Step | Notes | Notebooks |
|---|---|---|---|
| 0 | Data | [`00_data.md`](00_data.md): dataset, download, train/valid/test split | [`00_download_data`](../notebooks/00_download_data.ipynb) |
| 1 | Exploratory data analysis | [`01_eda.md`](01_eda.md): counts, Live/Dead, cell size, blur, overlap, colour | [`01_eda_overview`](../notebooks/01_eda_overview.ipynb) |
| 2 | Baseline models (B0–B2) | [`02_baseline_results.md`](02_baseline_results.md): scores and figures<br>[`02_baseline_explained.md`](02_baseline_explained.md): plain-language guide to each model and metric | [`02_baseline_cellpose`](../notebooks/02_baseline_cellpose.ipynb) |
| 3 | Our model: YOLO26n training and augmentation | [`03_yolo_training.md`](03_yolo_training.md): training recipe, validation metrics, augmentation screen<br>[`03a_rgb_grayscale_results.md`](03a_rgb_grayscale_results.md): RGB vs grayscale input<br>[`03a_rgb_grayscale_diagnostics.md`](03a_rgb_grayscale_diagnostics.md): per-source errors and example images | [`03_initial_model_train`](../notebooks/03_initial_model_train.ipynb), [`03a_rgb_grayscale`](../notebooks/03a_rgb_grayscale.ipynb), [`03b_augmentation_exploration`](../notebooks/03b_augmentation_exploration.ipynb) |
| 4 | Our model vs the baseline | [`04_yolo_performance.md`](04_yolo_performance.md): count error 1.6 vs 5.6 cells per image | [`04_yolo_performance`](../notebooks/04_yolo_performance.ipynb) |
| 5 | Extra experiments: other datasets | [`05_extra_experiments.md`](05_extra_experiments.md): synthetic cells (BBBC005); blood cells, tomatoes, eggs | [`05a_model_application`](../notebooks/05a_model_application.ipynb), [`05b_extra_experimentation`](../notebooks/05b_extra_experimentation.ipynb) |

## Folders

```
figures/
  eda/                figures in 01_eda (copied from the notebook outputs)
  baseline/           figures in 02_baseline_results
  rgb_grayscale/      figures in 03a_rgb_grayscale_*
  yolo_training/      figures in 03_yolo_training
  yolo_performance/   figures in 04_yolo_performance
  extra_experiments/  figures in 05_extra_experiments
  slides/             slide versions of the 02_baseline figures (same data, dark slide theme), plus make_slide_figs.py, which regenerates them
results/
  rgb_grayscale/      CSV and JSON results behind 03a_rgb_grayscale_*
```

## Slide figures

The figures used in our slides are in [`figures/slides/`](figures/slides/). They show the same results as the figures in the notes, redrawn in the slides' dark theme. [`make_slide_figs.py`](figures/slides/make_slide_figs.py) regenerates them from the outputs of `02_baseline_cellpose.ipynb`.

## Adding a note

Name it `<step number>_<topic>.md` (or `<notebook number>_<topic>.md` for a note on one sub-notebook, like `03a_`), copy the header line from an existing note, and add it to the table above.
