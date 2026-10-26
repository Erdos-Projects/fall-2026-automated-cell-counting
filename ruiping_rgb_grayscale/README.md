# RGB and Grayscale Cell Counting

Contributor: Ruiping Huang.

This contribution compares RGB and three-channel grayscale inputs for total cell counting with YOLO26n. Live and Dead annotations are merged into one Cell class while retaining all boxes.

The completed seed-0 run favored RGB: test count MAE was **1.173 cells/image** for RGB and **1.587** for grayscale on the same 75 images. See the [results report](docs/results_report.md) for the comparison and limitations.

For a short overview, start with the [one-page Week 5 check-in PDF](docs/weekly_checkin.pdf) or its [Markdown source](docs/weekly_checkin.md). The [source and image diagnostics](docs/diagnostics.md) include six box comparisons and source-level errors. The [team notebook comparison](docs/team_notebook_comparison.md) records which settings were shared and which differed.

The [meeting script and evidence guide](docs/meeting_guide.md) follows the talking points with direct links to the supporting tables, figures, configuration, and code.

## Experiment

- Source: Roboflow `ruiping-huang/cell-counting-zeqo5-ncyud`, Version 1, YOLO26 export.
- Data: 500 unique source images; the team manifest fixes 351 training, 74 validation, and 75 test images.
- Arms: RGB and grayscale, initialized independently from the same pretrained checkpoint.
- Shared settings: 640-pixel inputs, seed 0, batch size 16, up to 100 epochs, and patience 20.
- Training augmentation: horizontal/vertical flips and right-angle rotations only. Other random augmentation, including optional library defaults, is disabled.
- Selection: validation count MAE determines each confidence threshold. Test results do not select model settings.
- Primary endpoint: total-count MAE. Secondary endpoints: RMSE, R-squared, bias, counts within 10%, and runtime.

## Files

| File | Purpose |
|---|---|
| [`notebooks/rgb_grayscale_ablation.ipynb`](notebooks/rgb_grayscale_ablation.ipynb) | Self-contained download, data validation, training, evaluation, and diagnostics |
| [`src/api_dataset_alignment.py`](src/api_dataset_alignment.py) | Source of the manifest alignment functions embedded in the notebook |
| [`docs/experiment_plan.md`](docs/experiment_plan.md) | Design, shared settings, endpoints, and limitations |
| [`docs/decision_log.md`](docs/decision_log.md) | Decisions and verified preparation milestones |
| [`docs/results_report.md`](docs/results_report.md) | Completed experiment results, interpretation, and verification |
| [`results/`](results/) | Comparison, paired predictions, training curves/history, validation grids, and provenance |
| [`splits/splits.csv`](splits/splits.csv) | Local copy of the team's frozen image-level manifest |
| [`requirements.txt`](requirements.txt) | Direct dependencies for the experiment |
| [`scripts/build_notebook.py`](scripts/build_notebook.py) | Rebuilds the notebook with cleared outputs |
| [`scripts/analyze_completed_run.py`](scripts/analyze_completed_run.py) | Source summaries, verified supplementary inference, and illustrative box comparisons; no training |
| [`scripts/build_checkin_pdf.py`](scripts/build_checkin_pdf.py) | Render the check-in Markdown as a one-page PDF |

The notebook requires a CUDA runtime and reads `ROBOFLOW_API_KEY` from a Colab secret or a hidden prompt. It saves checkpoints and results to Google Drive. The download and training APIs follow the team's [YOLO notebook](https://github.com/Erdos-Projects/fall-2026-automated-cell-counting/blob/main/notebooks/03_cell_counting.ipynb).

Each run records dependency versions, hardware, the manifest commit and hash, image alignment, shared settings, actual training arguments, initial and fitted checkpoint hashes, and timestamped stage events. Credentials are not written to these records.

## Status

Completed. RGB trained for 100 epochs; grayscale stopped after 39 epochs under the shared patience-20 rule, with its best checkpoint from epoch 19. Both arms were evaluated on the same held-out test set. Data, augmentation, prediction, argument, and artifact-integrity checks passed. Earlier local one-epoch checks used an earlier configuration and are not reported as performance results.

## Scope

One seed and a random split do not establish generalization to unseen acquisition sessions. Pixel and coordinate equality with the team's original export remain unverified. Runtime comparisons require matched hardware and measurement scope.

Raw dataset images, credentials, model checkpoints, caches, and full runtime archives are excluded from source control. The `results/` directory contains the reviewed metrics, curves, and provenance. Earlier numbered notebooks, local operation history, and full execution archives are outside this contribution.

The diagnostic figures contain six annotated test-image examples. Original dataset directories remain outside this contribution.

## Reproduce the supplementary diagnostics

Install the listed dependencies and provide the completed run directory plus the prepared RGB dataset. The dataset must contain `test/images` and `test/labels`; decoded image hashes and counts are checked against the run inventory. The run directory must contain `image_alignment.csv` and both arms' `weights/best.pt`.

```bash
python scripts/analyze_completed_run.py \
  --run-root /path/to/completed/run \
  --images-root /path/to/prepared/rgb \
  --reference-provenance "Describe the annotation export and any coordinate verification limitations"
```

The script uses the existing checkpoints and confidence thresholds, verifies that all 75 counts per arm match the saved evaluation, and writes only supplementary diagnostics. It does not train or tune. After a verified first run, `--reuse-boxes` regenerates figures from the saved selected-case boxes without inference.
