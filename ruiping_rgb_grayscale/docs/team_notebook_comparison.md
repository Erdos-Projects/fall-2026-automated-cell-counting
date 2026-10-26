# Comparison with the team YOLO notebook

The RGB arm uses the same model family as the team's training notebook. It is a newly fitted control for the RGB/grayscale experiment, not the teammate's fitted model and not an exact replication of that training setup.

The inspected source is [notebooks/03_cell_counting.ipynb](https://github.com/Erdos-Projects/fall-2026-automated-cell-counting/blob/main/notebooks/03_cell_counting.ipynb), Git blob `19f704472f1f1641cf73ccd8b09eaf69621b794c`, inspected 2026-10-26. The entries below come from its code and saved training output; they do not describe unshared teammate experiments.

| Item | Inspected team notebook | Completed RGB control |
|---|---|---|
| Architecture and initialization | `YOLO("yolo26n.pt")`; pretrained initialization | Same model/checkpoint identifier; independent training |
| Initial checkpoint bytes | Hash not recorded in notebook | Hash recorded and matched across the RGB/grayscale arms |
| Task | `single_cls=True` | Live and Dead merged to Cell; total count |
| Image size / batch / training seed | 640 / 16 / 0 in training log | 640 / 16 / 0 |
| Training runtime | Colab Tesla T4; Ultralytics 8.4.183 | Colab Tesla T4; Ultralytics 8.4.183 |
| Source | Pranjal workspace, project `cell-counting-zeqo5-raras`, Version 1, YOLO26 | Ruiping workspace, project `cell-counting-zeqo5-ncyud`, Version 1, YOLO26 |
| Split handling | Trains directly from downloaded `data.yaml`; logs 350 training and 100 validation images | Reconstructs manifest membership: 351 training, 74 validation, 75 test |
| Maximum epochs / patience | 100 / 100 in training log | 100 / 20; RGB completed 100 |
| Horizontal / vertical flips | 0.5 / 0.0 | 0.5 / 0.5 |
| Random right-angle rotation | Not specified | Uniform 0, 90, 180, 270 degrees |
| HSV jitter | 0.015 / 0.7 / 0.4 | All zero |
| Scale / translation | 0.5 / 0.1 | Both zero |
| Mosaic | 1.0; disabled for last 10 epochs | Zero throughout |
| Optional Albumentations | Logs Blur, MedianBlur, ToGray, CLAHE, each probability 0.01 | Replaced with the specified rotation transform |
| Evaluation shown | Detector training and validation output | Validation count threshold selection; held-out per-image count MAE, RMSE, R-squared, tolerance rate, bias, timing, and diagnostics |

The team notebook relies on library defaults for several settings. The log, rather than only the short `model.train(...)` call, reveals these values. The RGB and grayscale arms in this contribution share the revised augmentation and stopping settings with each other.

Dataset names and image counts do not prove identical image pixels or annotation coordinates across workspace exports. The team notebook also does not record the initial checkpoint hash. These limitations prevent a byte-for-byte equivalence claim.

Because split membership, augmentation, and stopping settings differ, any score difference between the historical team run and this RGB run cannot be assigned to one change. The RGB/grayscale comparison within the completed paired experiment is the relevant color-processing comparison. Detection mAP and total-count accuracy are different metrics.
