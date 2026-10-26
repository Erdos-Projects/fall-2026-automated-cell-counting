# Week 5 cell counting check-in

Ruiping Huang | RGB and grayscale preprocessing comparison

**Result.** RGB achieved lower total-cell counting error than grayscale in this seed-0 YOLO26n experiment. Grayscale did not improve the tested configuration.

**Question and preparation.** The team EDA raised background color as a possible confounder. I tested grayscale conversion using 500 Roboflow Version 1 originals, with the team's frozen 351/74/75 train/validation/test membership. All 26,715 Live and Dead boxes were retained as one Cell class. Grayscale luminance was replicated into three channels; image geometry and paired labels were unchanged.

**Training and evaluation.** Both models started from the same official pretrained checkpoint, using 640-pixel inputs, batch 16, seed 0, horizontal/vertical flips, and right-angle rotations. Other random augmentation was disabled. Both used a 100-epoch maximum and patience 20 on a Colab T4. Existing validation-selected confidence thresholds, RGB 0.44 and grayscale 0.33, were applied to the same 75 test images.

| Test metric | RGB | Grayscale |
|---|---:|---:|
| Count MAE, cells/image | 1.173 | 1.587 |
| Count RMSE, cells/image | 1.592 | 2.248 |
| R-squared | 0.9885 | 0.9770 |
| Within 10% of reference | 75/75 | 71/75 |
| Largest absolute count error | 4 | 9 |

The within-10% measure is a tolerance rate. Exact totals occurred on 22 RGB and 19 grayscale images.

**What worked and what did not.** Both models counted all 56 cells in one reviewed example. Grayscale had higher MAE in three acquisition sources and equal MAE in two; the largest gap was source 3_A, 0.733 versus 1.933. The largest grayscale error was reference 65, RGB 61, grayscale 56. An RGB-exact case still had unmatched reference and prediction boxes that canceled in the total, showing why count metrics need visual diagnostics.

**Limitations and next step.** RGB completed 100 epochs; grayscale stopped at 39, with its best checkpoint from 19. Checkpoints and stopping used validation detection mAP50-95, not count MAE. This is one seed with all sources represented in every split. Post-hoc examples do not establish a color mechanism. I propose reviewing ambiguous/border annotations and agreeing on detection evaluation, then choosing repeated seeds or an equal-budget study. No supplementary training has started.

**Team context.** This retains the team's YOLO26n architecture but changes the frozen split, augmentation, stopping patience, and count evaluation relative to the inspected team notebook. It is not a direct replication. The project Word document reports historical Cellpose MAE 5.6; that run was not reproduced here, and its evaluation compatibility requires checking before a formal comparison.

Evidence: [full report](results_report.md), [source and image diagnostics](diagnostics.md), and [team notebook comparison](team_notebook_comparison.md).
