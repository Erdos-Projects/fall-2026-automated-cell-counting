# Cell Image Counting

> **Overview** · [All notes](README.md)

## Scope

Counting cells in microscopy images is still mostly done by hand, which is slow and tedious. We are building a deep-learning tool that takes a microscopy image and outputs the cell count. It finds each cell, then counts what it found, so every count can be checked against the image. We want it to work on images from different cell sources and imaging conditions. We test it explicitly on the hard cases: high cell density, overlapping cells, and poor image quality such as blur.

## Data sources

| Dataset | Status | Description |
|---|---|---|
| [Roboflow Cell Counting](https://universe.roboflow.com/cell-counting/cell-counting-zeqo5) | **In use** | 500 brightfield images (640×640) from 5 acquisition sessions, with 23–91 cells each, boxed as Live or Dead. See [`00_data.md`](00_data.md). |
| Synthetic Cell Images and Masks (BBBC005) | Used in extra experiments ([`05_extra_experiments.md`](05_extra_experiments.md)) | 20,400 synthetic images at known blur levels, labelled with cell counts |
| CoNIC | Not used; a candidate for expanding the project | Histology nuclei: dense, crowded scenes |

## Stakeholders

- **Pathology labs and medical companies:** automated counts for diagnostics and quality control.
- **Research labs:** cell-culture monitoring and experiments that depend on counts.

## KPIs

| KPI | Measure | Baseline (Cellpose-SAM) | **Our model (YOLO26n)** |
|---|---|---|---|
| Count accuracy | Mean absolute error in cells per image; share of images within ±10% | 5.6 cells; 59% | **1.6 cells; 96%** |
| R² | Variance in true counts that the predictions explain | 0.74 | **0.98** |
| Compute | Time per image, compared with counting by hand (about 1 minute) | 1.8 s | **28.6 ms** (about 63× faster) |

All on the same 75 test images. Details: [`02_baseline_results.md`](02_baseline_results.md), [`04_yolo_performance.md`](04_yolo_performance.md). Roadmap: [`plan.md`](plan.md).
