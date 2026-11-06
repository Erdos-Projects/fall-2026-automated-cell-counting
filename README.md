# Automated Cell Image Counter with Deep Learning

Can a small deep-learning model count cells in microscopy images faster and more accurately than counting by hand or a large general-purpose model?

We take a brightfield microscopy image and return the number of cells in it, with a box around every cell counted so each count can be checked. Scope and KPIs are in [`notes/project.md`](notes/project.md), the weekly roadmap is in [`notes/plan.md`](notes/plan.md), and all write-ups are listed in the [notes index](notes/README.md). The slides from our final presentation are in [`slides/Erdos-DL-cell-image-counter.pdf`](slides/Erdos-DL-cell-image-counter.pdf).

## Contributors

- **Dingshan Deng**
- **Pranjal Singh**
- **Jungmin Seo**
- **Ruiping Huang**
- **Mentors: Lindsay Warrenburg & Marcos Ortiz**

---

## Abstract

Pathology labs, medical companies and research labs count cells routinely, for diagnostics, quality control and cell-culture experiments. Counting by hand takes about a minute for a typical image of about 50 cells, and longer for crowded ones. It is repetitive, and different people can count the same image differently. We built a deep-learning tool that takes a microscopy image and returns the cell count, together with a box around every cell it counted, so each count can be checked against the image. We used 500 annotated brightfield images from five acquisition sessions (Roboflow Universe), with a shared train/validation/test split. As the baseline we used Cellpose-SAM, a large general-purpose cell-segmentation model, with its thresholds tuned on the training images. Our model, the YOLO26n object detector fine-tuned on our labelled cells, counts the 75 held-out test images with a mean absolute error of 1.6 cells per image (R² = 0.98, 96% of counts within ±10%). The tuned baseline is off by 5.6 cells (R² = 0.74), and by 34.8 cells with its default settings. Our model also takes 28.6 ms per image, against 1.8 s for Cellpose-SAM, about 63 times faster. Training a small model on the task at hand beat a much larger general-purpose one. Applied without retraining to synthetic fluorescence images, the model counts small, round, separated cells almost exactly, but misses many large, irregular, touching cells. Therefore, the small model we built is a specialized model, not a general-purpose one. Cell images vary widely in cell type, staining, magnification and microscope, and both results show that a single general-purpose model does not handle this range well. We expect labs to get the most from the approach by fine-tuning the model on a few hundred labelled images from their own setup.

---

## Results

All models are scored on the same 75 held-out test images of [`splits/splits.csv`](splits/splits.csv) (about 53 cells per image).

| Model | Count error (MAE, cells / image) | Within ±10% | R² | Time per image |
|---|---|---|---|---|
| B0 session mean (reference) | 6.9 | 45% | 0.71 | — |
| B1 Cellpose-SAM, default settings | 34.8 | 9% | −6.21 | 1.8 s |
| B2 Cellpose-SAM, tuned on train (**baseline**) | 5.6 | 59% | 0.74 | 1.8 s |
| **YOLO26n, fine-tuned (ours)** | **1.6** | **96%** | **0.98** | **28.6 ms** |

- Our model is about 3.5 times more accurate than the baseline and about 63 times faster. Counting by hand takes about a minute for an image of about 50 cells.
- It finds 98% of the labelled cells, and 97% of its boxes are real cells (validation images, IoU ≥ 0.5).
- The notebooks run locally or on Google Colab. Our model's inference time was measured on a Google Colab Tesla T4 GPU.

Details: baselines in [`notes/02_baseline_results.md`](notes/02_baseline_results.md), our model vs the baseline in [`notes/04_yolo_performance.md`](notes/04_yolo_performance.md), and tests on other datasets in [`notes/05_extra_experiments.md`](notes/05_extra_experiments.md).

---

## Data

- **Primary:** [Roboflow Cell Counting](https://universe.roboflow.com/cell-counting/cell-counting-zeqo5): 500 brightfield images (640×640) from 5 acquisition sessions, with 23–91 cells each, every cell boxed as Live or Dead. We use the raw export of our Roboflow fork, not the public version, whose ~1,200 images are the same 500 plus augmented copies. See [`notes/00_data.md`](notes/00_data.md).
- **Split:** [`splits/splits.csv`](splits/splits.csv) assigns each image to train / validation / test (351 / 74 / 75), stratified by session and cell count so every session and density range appears in each split. Every model uses it.
- **Other datasets** (testing the model without retraining): [Synthetic Cell Images and Masks (BBBC005)](https://www.kaggle.com/datasets/vbookshelf/synthetic-cell-images-and-masks-bbbc005-v1), and Roboflow blood-cell, tomato and egg datasets.
- Images are downloaded into `data/`, which is gitignored.

---

## Start here

1. **Set up** (local): `conda env create -f environment.yml`, then copy `.env.example` to `.env` and fill in `ROBOFLOW_EXPORT_URL`. On Colab, add the same name under Secrets. Notebooks 00, 02 and 04 use this repo's `cellcount` package. The conda environment installs it locally; for notebooks 02 and 04, also run `pip install -e ".[dev,model]"` for the modelling extras (torch, ultralytics, cellpose). On Colab, their first cell installs it, with the extras. The other notebooks run on Colab and install what they need (`roboflow`, `ultralytics`) in their first cells.
2. **Run the notebooks in order:**

| Notebook | What it does | Notes |
|---|---|---|
| [`00_download_data.ipynb`](notebooks/00_download_data.ipynb) | Downloads the 500 Roboflow images into `data/` | [`notes/00_data.md`](notes/00_data.md) |
| [`01_eda_overview.ipynb`](notebooks/01_eda_overview.ipynb) | Exploratory data analysis: counts, Live/Dead, image quality, overlap, colour | [`01_eda`](notes/01_eda.md) |
| [`02_baseline_cellpose.ipynb`](notebooks/02_baseline_cellpose.ipynb) | Baseline: off-the-shelf Cellpose-SAM counts cells (GPU recommended) | [`02_baseline_results`](notes/02_baseline_results.md), [`02_baseline_explained`](notes/02_baseline_explained.md) |
| [`03_initial_model_train.ipynb`](notebooks/03_initial_model_train.ipynb) | Our model: YOLO26n trained on our images | [`03_yolo_training`](notes/03_yolo_training.md) |
| [`03a_rgb_grayscale.ipynb`](notebooks/03a_rgb_grayscale.ipynb) | YOLO26n trained on RGB vs grayscale input | [`03a_rgb_grayscale_results`](notes/03a_rgb_grayscale_results.md), [`03a_rgb_grayscale_diagnostics`](notes/03a_rgb_grayscale_diagnostics.md) |
| [`03b_augmentation_exploration.ipynb`](notebooks/03b_augmentation_exploration.ipynb) | Screens training augmentations one at a time | [`03_yolo_training`](notes/03_yolo_training.md) |
| [`04_yolo_performance.ipynb`](notebooks/04_yolo_performance.ipynb) | Our model vs the baseline on the shared test split | [`04_yolo_performance`](notes/04_yolo_performance.md) |
| [`05a_model_application.ipynb`](notebooks/05a_model_application.ipynb) | Applies our model to synthetic cell images (BBBC005) | [`05_extra_experiments`](notes/05_extra_experiments.md) |
| [`05b_extra_experimentation.ipynb`](notebooks/05b_extra_experimentation.ipynb) | Applies our model to other datasets (blood cells, tomatoes, eggs) | [`05_extra_experiments`](notes/05_extra_experiments.md) |

---

## Layout

```
notebooks/        notebooks that run the analysis
src/cellcount/    reusable code (data loading, EDA, splits, metrics, models)
splits/           splits.csv: the shared train/valid/test split, which every model must use
notes/            write-ups numbered like their notebooks; start at notes/README.md
                  (slide figures: notes/figures/slides/, the note figures in the slide theme)
slides/           slides from the final presentation (PDF)
data/, outputs/   gitignored: downloaded images, results, model caches
```
