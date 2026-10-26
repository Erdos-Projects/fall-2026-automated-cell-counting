# RGB and Grayscale Ablation Plan

Status: protocol approved on 2026-10-25 and completed on 2026-10-26 UTC. RGB trained for 100 epochs; grayscale stopped at epoch 39 under the shared early-stopping rule. Both arms completed the prespecified held-out evaluation. See [the results report](results_report.md).

## Research question

How does replacing RGB input with three-channel grayscale input affect total cell-count error under otherwise matched training conditions?

## Data and target

Use Version 1 of `ruiping-huang/cell-counting-zeqo5-ncyud`. The project page shows 500 images, a 350/100/50 export split, and no additional version preprocessing or augmentation. Version 2 contains 1,180 images with flips and right-angle rotations and is excluded from this experiment.

Download the YOLO26 export and reconstruct the team's fixed 351/74/75 split from `splits/splits.csv`, pinned to commit `b646a2eadb6ea4419e87c8f65b4c108d437a8861`. Match unique original filenames and check each image's annotation count. Retain every box and map both Live and Dead to Cell.

Reject changed image identities, duplicate originals, augmented copies, incompatible image dimensions, missing labels, count mismatches, and invalid box coordinates. Resolve discrepancies before training.

## Matched arms

| Setting | RGB | Grayscale |
|---|---|---|
| Image processing | Decoded RGB saved as lossless PNG | Luminance replicated across three channels, saved as lossless PNG |
| Images and boxes | Frozen manifest | Same manifest and boxes |
| Model | YOLO26n | YOLO26n |
| Initialization | Official pretrained checkpoint | Same checkpoint hash, independent run |
| Image size | 640 | 640 |
| Class target | Cell | Cell |
| Seed | 0 | 0 |
| Batch size | 16 | 16 |
| Maximum epochs | 100 | 100 |
| Early stopping patience | 20 | 20 |
| Random augmentation | Flips and right-angle rotations only | Same rules |

Use minimal orientation augmentation to preserve cell appearance and avoid additional color or contrast changes during the RGB/grayscale comparison. It applies only to the 351 training images and does not create additional dataset members. Both arms use these settings; this experiment is not an exact reproduction of the team's earlier YOLO run.

### Explicit augmentation settings

| Argument | Value | Meaning |
|---|---:|---|
| `fliplr` | 0.5 | Horizontal flip probability |
| `flipud` | 0.5 | Vertical flip probability |
| Custom `RandomRotate90` | `p=1.0` | Uniform 0/90/180/270-degree rotation; identity probability 0.25 |
| `degrees` | 0.0 | Disable native continuous-angle rotation |
| `translate`, `scale`, `shear`, `perspective` | 0.0 | Disabled |
| `hsv_h`, `hsv_s`, `hsv_v`, `bgr` | 0.0 | Color, brightness, and channel-swap augmentation disabled |
| `mosaic`, `mixup`, `cutmix`, `copy_paste` | 0.0 | Disabled |
| `close_mosaic` | 0 | No mosaic phase |

`albumentations==2.0.8` supplies the discrete rotation and transforms the bounding boxes with the image. Passing a custom transform list replaces optional blur, random grayscale, CLAHE, and other Albumentations defaults. Record and verify the initialized pipeline before the first training epoch; stop if the required transform is absent or additional transforms appear. Square 640-pixel images require no new rotation borders. Do not use Roboflow's offline three-outputs-per-image version.

The two flip decisions and rotation are sampled during training. Settings remain fixed across arms. A shared seed aids reproducibility but does not guarantee identical outputs across different hardware or software environments.

Validation and test preprocessing consists of deterministic image loading, RGB or three-channel luminance conversion according to the arm, YOLO channel/tensor formatting, resizing or padding to the configured input size if needed, and normalization from 0-255 to 0-1. The verified source images are already square 640-pixel images. No random augmentation or test-time augmentation is used for evaluation.

Implementation references: [Ultralytics custom Albumentations integration](https://docs.ultralytics.com/integrations/albumentations) and [RandomRotate90](https://albumentations.ai/explore/transform/RandomRotate90/docs/).

## Evaluation

Train both arms before evaluating the test set. Independently select the confidence threshold minimizing count MAE over the validation images, using a fixed grid from 0.01 to 0.90 in steps of 0.01. Report the same 75 test images for both arms.

- Primary: total-count MAE, in cells per image.
- Secondary: RMSE, R-squared, signed bias, fraction within 10%, and runtime.
- Diagnostics: training curves, predicted versus reference counts, and largest per-image errors.

Runtime includes image loading, model preprocessing, inference, and postprocessing after validation inference. Model loading and offline grayscale conversion are excluded and should be reported separately if needed.

## Records

Save `run_metadata.json`, `events.jsonl`, `splits.csv`, `alignment_metadata.json`, `image_alignment.csv`, each arm's `training_record.json`, `args.yaml`, `results.csv`, and `weights/best.pt`. Save validation threshold tables, selected thresholds, test counts, the comparison table, and diagnostic figures.

After evaluation, inventory the output files with SHA256 hashes and export a standalone ZIP archive. Runtime archives and checkpoints remain outside the source contribution.

## Execution stages

After approval, verify the CUDA/T4 runtime and dependencies, then read the Roboflow credential from a Colab secret or a hidden prompt. Download and validate Version 1, reconstruct the frozen split, and check paired RGB/grayscale data. Verify augmentation geometry and box preservation before training. Train RGB and grayscale sequentially, then select thresholds on validation and evaluate the held-out test set. Summarize the paired results and limitations in English.

Stop for review if source identity, annotations, split membership, runtime compatibility, or required training settings cannot be verified. Do not silently substitute data, a different model, a smaller batch size, or CPU execution. Keep the source contribution in `ruiping_rgb_grayscale/`, runtime outputs in Google Drive, and credentials outside saved files. Remote publication and merging are separate actions requiring authorization; merging also requires team agreement.

## Interpretation

The direction of the grayscale effect is unknown in advance. A single seed and the random split provide an initial comparison. Repeated seeds and leave-one-session-out evaluation are follow-up experiments. Identity/count matching does not prove equivalence to an unavailable original team export. Do not tune this experiment after inspecting test errors.
