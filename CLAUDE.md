# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Cell Image Counting: a deep learning model that takes a microscopy image and outputs an estimated cell count. The goal is generalization across cell sources and imaging conditions, with explicit evaluation on high cell density, overlapping cells, and poor image quality (blur). Project docs live in `notes/`: `notes/project.md` has the scope, data sources, stakeholders, and KPIs; `notes/plan.md` has the phased roadmap, metric definitions, and open team decisions. Check `plan.md` before proposing next steps, and put new markdown docs in `notes/`.

This is a shared repo. `CLAUDE.md` and `.claude/` are gitignored on purpose, so never stage them.

- **Primary dataset:** our Roboflow fork's raw COCO export: 500 images at 640×640, with Live and Dead boxes, from 5 acquisition sources (`2022_12_7_{1,2,3}_{A,B}`, parsed from file names into `source` and `frame`). The public version's ~1,200 images are the same images with augmented copies, so never use it. Details are in `data/README.md`.
- **Candidate datasets for later:** Synthetic Cell Images and Masks (blur levels) and CoNIC (histology nuclei). These cover the dense or overlapping case, which the Roboflow set lacks.
- **KPIs:** count accuracy and R², plus compute cost compared with counting by hand. `notes/plan.md` defines the metrics.

## Commands

```bash
conda env create -f environment.yml          # env "cellcount" (Python 3.12, same as Colab); installs the package with pip -e .[dev]
conda env update -f environment.yml --prune  # after changing dependencies in pyproject.toml
conda run -n cellcount jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=cellcount notebooks/01_eda.ipynb   # run a notebook headless
```

There are no tests or linter yet. Dependencies live in `pyproject.toml`, not `environment.yml`. Use lower bounds only, so Colab reuses its preinstalled packages; don't add torch until modeling starts.

## Architecture

- `src/cellcount/` holds all reusable logic; notebooks only orchestrate and plot. Keep it importable without torch.
  - `config.py`: `is_colab()`, `get_paths()` (the data/outputs root is the repo locally and `/content/drive/MyDrive/cell-image-counting` on Colab, or `$CELLCOUNT_STORAGE` if set), and `get_secret()` (Colab Secrets → env → `.env`).
  - `data.py`: `download_export()` (signed Roboflow URL, the default) and `download_roboflow()` (API; the fork has no generated version yet). `load_coco()` returns `(images, boxes)` DataFrames keyed by `uid = "<split>/<image_id>"` and drops Roboflow's placeholder `cell` category.
  - `eda.py`, `viz.py`: statistics (image colour and blur, phash, IoU overlap, background clusters) and plotting.
  - `splits.py`: `make_splits()` (StratifiedGroupKFold with 20 folds → 70/15/15) and `save_splits()`.
- `notebooks/NN_name.ipynb`: each starts with the same **bootstrap cell** (on Colab: mount Drive, clone or pull the repo using `BRANCH` and the `GH_TOKEN` secret, `pip install -e`). Copy it verbatim into new notebooks.
- `splits/splits.csv` is the committed, shared train/valid/test assignment (columns include `source`, so leave-one-source-out cross-validation can be built from it). Models must use it instead of Roboflow's split.

## Current state

Phases 0–1 are done on branch `v0.1.0-beta-Initial_EDA`. The findings are at the end of `notebooks/01_eda.ipynb`. Next is Phase 2 (baselines). The collaborator's original `cell_counting.ipynb` is still a skeleton; leave it alone.

## Conventions

- Branches are named by version and phase (e.g. `v0.1.0-beta-Initial_EDA`); PRs target `main`.
- `data/`, `outputs/`, `models/`, `checkpoints/`, and weight files are gitignored (`data/README.md` is tracked). Secrets (`ROBOFLOW_EXPORT_URL`, `ROBOFLOW_API_KEY`, `GH_TOKEN`) go in `.env` or Colab Secrets. Never write them into notebooks or notes.
