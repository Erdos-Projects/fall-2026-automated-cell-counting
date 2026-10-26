# Meeting script and evidence

Ruiping Huang | Week 5 RGB and grayscale cell counting

Follow the short script below and open the evidence beside each point. Start with the [one-page check-in](weekly_checkin.pdf), or read its [Markdown version](weekly_checkin.md).

| Talking point | Jump to script |
|---|---|
| Completed work and motivation | [What I completed](#what-i-completed) |
| Differences from the previous team notebook | [Experiment differences](#experiment-differences) |
| Test results and source analysis | [Counting results](#counting-results) |
| Image-level finding | [Box comparisons](#box-comparisons) |
| Limitations and proposed next step | [Next step](#next-step) |

## What I completed

This week, I completed an RGB-versus-grayscale experiment for cell counting using YOLO26n on a Colab T4 GPU. The goal was to test our earlier idea that removing color might reduce the influence of background differences.

Evidence: [experiment and runtime](results_report.md#experiment), [recorded configuration](../results/run_metadata.json), and [training and evaluation notebook](../notebooks/rgb_grayscale_ablation.ipynb).

## Experiment differences

I kept the same model architecture as the team's previous YOLO work. Compared with the notebook I reviewed, I reconstructed the team's fixed 351/74/75 split, restricted augmentation to flips and right-angle rotations, and added count-based evaluation on the held-out test set. Both live and dead cells were counted together.

Evidence: [side-by-side settings comparison](team_notebook_comparison.md), [frozen split](../splits/splits.csv), and [actual RGB training arguments](../results/rgb/training_record.json).

The RGB arm is the matched control for this grayscale experiment. Differences from the historical team run involve several settings, so their effects cannot be separated by comparing the two runs' scores.

## Counting results

On the same 75 test images, RGB achieved a mean absolute error of 1.17 cells per image, compared with 1.59 for grayscale. Grayscale had higher error in three acquisition sources and equal error in two. Grayscale did not improve counting under this setup.

Show the [test results table](results_report.md#held-out-test-results), [predicted versus reference count plot](figures/predicted_vs_reference.png), and [source MAE plot](figures/diagnostics/mae_by_source.png).

Supporting data: [paired per-image counts](../results/paired_test_counts.csv) and [source-level metrics](../results/diagnostics/per_source_metrics.csv). The source pattern does not establish background color as the cause.

## Box comparisons

The box comparisons showed that a correct total count can hide detection discrepancies. In one RGB-exact example, unmatched reference and prediction boxes canceled in the total. It is useful to inspect detections as well as counts.

Show [train_258](figures/diagnostics/rgb_exact_grayscale_under.png): reference 67, RGB 67, grayscale 62. Under the illustrative matching rule, RGB has two unmatched reference boxes and two unmatched predictions. See the [case explanation](diagnostics.md#exact-count-can-hide-detection-discrepancies) and [matching table](../results/diagnostics/illustrative_box_matches.csv).

Additional examples:

| Example | Reference / RGB / grayscale | Figure |
|---|---|---|
| Largest grayscale error | 65 / 61 / 56 | [train_177](figures/diagnostics/largest_grayscale_error.png) |
| Grayscale count closer to reference | 58 / 54 / 57 | [valid_67](figures/diagnostics/grayscale_better.png) |
| Shared overcount | 44 / 48 / 49 | [train_207](figures/diagnostics/both_overcount.png) |
| Shared undercount | 61 / 58 / 57 | [train_243](figures/diagnostics/both_undercount.png) |
| Both counts exact | 56 / 56 / 56 | [valid_7](figures/diagnostics/both_exact.png) |

These examples are test images despite their original-export filename prefixes. Matching uses one-to-one IoU at least 0.50. Unmatched boxes can reflect localization or annotation differences. Reference coordinates come from a local COCO-derived cache and have not been directly compared with the formal YOLO export. These are qualitative diagnostics, not a full test detection benchmark. [Verification](../results/diagnostics/verification.json) confirms matching image hashes and exact reproduction of all 75 original counts per arm using the saved checkpoints and thresholds.

## Next step

This was a single-seed experiment. RGB completed 100 epochs, while grayscale stopped early at 39 under the same rule. I would like to discuss the difficult cases with the team before choosing the next experiment.

Evidence: [training and stopping](results_report.md#training-and-validation-selection), [limitations](results_report.md#interpretation-and-limits), and [proposed follow-up](diagnostics.md#implications-for-the-next-iteration). No supplementary training has started.

## Code and reproducibility

The [contribution README](../README.md) describes the files and diagnostic reproduction command. The [experiment notebook](../notebooks/rgb_grayscale_ablation.ipynb) includes recorded GPU information, training logs, evaluation tables, and embedded figures from the completed run. Console formatting was normalized for readability, without changing numerical results or image outputs. The original unmodified execution archive, checkpoints, and dataset directories are retained separately. A Roboflow credential is supplied at runtime through a Colab secret or hidden prompt.
