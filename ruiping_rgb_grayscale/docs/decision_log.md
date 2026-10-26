# Experiment Decisions and Verification Record

Run: `rgb_grayscale_seed0_20261026T040157Z`. Status: completed on 2026-10-26 UTC. Both models, held-out evaluation, diagnostics, and artifact verification are complete.

## Scientific decisions

| ID | Decision | Rationale and evidence |
|---|---|---|
| D01 | Predict total Cell count | Combine Live and Dead while retaining every annotation box; consistent with the team objective and `single_cls=True`. |
| D02 | Use Roboflow Version 1 | 500 original images without additional version preprocessing or augmentation; Version 2 is excluded. |
| D03 | Reconstruct the frozen 351/74/75 split | Match image membership using the team's pinned manifest, not only split sizes. |
| D04 | Compare RGB with three-channel grayscale | Change color information while preserving geometry and labels. |
| D05 | Initialize both arms independently | Use the same official pretrained `yolo26n.pt`, with identical initial file hashes; neither arm starts from a fitted cell-counting checkpoint. |
| D06 | Use seed 0 | Match the shared helper and observed team training log; retain one seed as a stated limitation. |
| D07 | Disable color jitter in both arms | Avoid interactions between color perturbation and the RGB/grayscale comparison. |
| D08 | Select thresholds using validation only | Minimize validation count MAE over the fixed 0.01-0.90 grid; evaluate the same 75 test images without retuning. |
| D09 | Record runtime provenance | Retain dependency versions, actual arguments, checkpoint hashes, split identity, and timestamped events. |
| D11 | Use orientation-only augmentation | Horizontal and vertical flip probabilities are each 0.5; rotation is uniform over 0/90/180/270 degrees. Other random augmentation, including optional library defaults, is disabled. |

The earlier proposal to retain YOLO's default brightness, scale, translation, and mosaic augmentation was replaced by D11 before formal training. Both completed arms used D11.

Both arms used a maximum of 100 epochs and early-stopping patience 20. Different realized training lengths are permitted under this shared rule; the run is not a fixed equal-epoch budget comparison.

The versioned YOLO detection fitness is validation mAP50-95, not count MAE. Early stopping therefore does not guarantee that further training could not improve counting. The completed result compares a common stopping procedure. Different realized epoch counts do not by themselves invalidate that comparison. A fixed-budget experiment would answer the separate question of performance after the same number of training epochs. The proposed fixed-100-epoch supplement is on hold while the design is reviewed; no supplementary training has started.

## Initialization evidence

The team's notebook loads `YOLO("yolo26n.pt")` and its training log records pretrained initialization. The completed run downloaded the official Ultralytics checkpoint and required the same initial hash for both arms. The initial checkpoint SHA256 is `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`. The fitted team `best.pt` and Roboflow RF-DETR checkpoints were not used for initialization.

## Verified execution milestones

| Milestone | Verified result |
|---|---|
| API data source | Version 1 YOLO26 export; 500 originals and 26,715 boxes. |
| Frozen split | 351 train / 74 validation / 75 test, with all original identities and per-image annotation counts matched. |
| Manifest bytes | Pinned SHA256 `b6f0cae5b41154d5d4b35239e662667260393ac8c01bfb909d7e252fd3fdfeb0`; an earlier local copy differed only by a trailing blank line and was made byte-identical before this run. |
| Paired inputs | RGB/grayscale labels identical; all grayscale images have three equal channels. |
| Geometry preflight | 64 sampled rotations covered all four orientations and preserved image pixels and boxes; both forced flips passed. |
| Runtime | Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130, Ultralytics 8.4.183, Albumentations 2.0.8. |
| Actual settings | Actual training arguments matched across arms except dataset/output names and paths; the initialized augmentation pipeline was verified in both runs. |
| RGB training | 100 epochs; completed at 2026-10-26T04:17:50Z; best checkpoint saved. |
| Grayscale training | 39 epochs; patience-20 early stopping; best checkpoint from epoch 19; completed at 2026-10-26T04:23:30Z. |
| Validation selection | RGB confidence 0.44 and grayscale confidence 0.33, selected from 74 validation images. |
| Held-out evaluation | Same 75 test images; RGB MAE 1.173333 and grayscale MAE 1.586667 cells/image. No post-test retuning. |
| Independent result checks | Paired test identities and reference counts matched the frozen manifest; count metrics recomputed from saved predictions. |
| Artifact integrity | Local archive hash matched the remote archive; all 59 inventory file hashes and both fitted checkpoint hashes verified. |

Full preparation and account-connection history is retained in the local execution archive. Credentials were supplied at runtime and were not written into source or experiment records.

See [the results report](results_report.md) for the comparison, interpretation, and limitations, and [run metadata](../results/run_metadata.json) for the recorded configuration.

## Supplementary descriptive analysis on 2026-10-26

No additional training, confidence selection, or modification of the original evaluation was performed. Source-level metrics were calculated from the saved paired test counts joined to the frozen manifest. All 75 images were accounted for, and weighted source MAEs recovered the original overall results.

Six illustrative cases were selected using deterministic error-based rules, including a grayscale-favorable case and a shared successful case. All 75 locally cached decoded RGB image hashes matched the formal-run inventory. Both saved checkpoint hashes were verified, and CPU inference with the existing thresholds reproduced all 75 original counts per arm exactly. Reference overlay coordinates come from a local Version 1 COCO-derived cache; equality with the formal YOLO export coordinates has not been directly verified. IoU-0.50 box matching is illustrative and does not constitute a new full-test detection benchmark.

Grayscale source MAE was higher in three sources and equal in two. The largest difference was source 3_A, RGB 0.733 versus grayscale 1.933. An RGB-exact image had two unmatched references and two unmatched predictions, illustrating cancellation in total counts. These observations do not identify a causal mechanism or establish significance across training seeds.

The inspected team notebook, Git blob `19f704472f1f1641cf73ccd8b09eaf69621b794c`, used the same model identifier but different split handling, augmentation, stopping patience, and evaluation. These differences are recorded in the [team notebook comparison](team_notebook_comparison.md). The paired RGB arm is not described as an exact replication of that run.

The [Week 5 check-in](weekly_checkin.md) summarizes preprocessing, first-iteration performance, failure inspection, limitations, and a proposed next step. The fixed-100-epoch supplement remains on hold.
