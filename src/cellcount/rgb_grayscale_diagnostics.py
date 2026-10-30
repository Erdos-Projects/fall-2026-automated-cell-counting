"""Build source summaries and illustrative box diagnostics without training or tuning."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/"notes/results/rgb_grayscale"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_cases(frame):
    remaining = frame.copy()
    chosen = []
    rules = [
        ("largest_grayscale_error", "Largest grayscale count error", lambda x: x,
         lambda x: x.gray_abs),
        ("rgb_exact_grayscale_under", "RGB exact; grayscale undercount", 
         lambda x: x[(x.rgb_error == 0) & (x.grayscale_error < 0)], lambda x: x.gray_abs),
        ("grayscale_better", "Grayscale closer to reference", 
         lambda x: x[x.rgb_abs > x.gray_abs], lambda x: x.rgb_abs - x.gray_abs),
        ("both_overcount", "Both models overcount", 
         lambda x: x[(x.rgb_error > 0) & (x.grayscale_error > 0)],
         lambda x: x.rgb_abs + x.gray_abs),
        ("both_undercount", "Both models undercount", 
         lambda x: x[(x.rgb_error < 0) & (x.grayscale_error < 0)],
         lambda x: x.rgb_abs + x.gray_abs),
        ("both_exact", "Both counts exact", 
         lambda x: x[(x.rgb_error == 0) & (x.grayscale_error == 0)],
         lambda x: -(x.reference_count - frame.reference_count.mean()).abs()),
    ]
    for category, title, subset, score in rules:
        candidates = subset(remaining).copy()
        if candidates.empty:
            raise ValueError(f"No candidate for {category}")
        candidates["selection_score"] = score(candidates)
        row = candidates.sort_values(["selection_score", "uid"], ascending=[False, True]).iloc[0]
        selected = row.to_dict()
        selected.update(category=category, title=title)
        chosen.append(selected)
        remaining = remaining[remaining.uid != row.uid]
    return pd.DataFrame(chosen)


def reference_boxes(path):
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    boxes = []
    for _, cx, cy, w, h in rows:
        boxes.append([(cx-w/2)*640, (cy-h/2)*640, (cx+w/2)*640, (cy+h/2)*640])
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def match_boxes(reference, prediction, threshold=0.5):
    if not len(reference) or not len(prediction):
        return set(), set()
    top_left = np.maximum(reference[:, None, :2], prediction[None, :, :2])
    bottom_right = np.minimum(reference[:, None, 2:], prediction[None, :, 2:])
    intersection = np.maximum(bottom_right-top_left, 0).prod(axis=2)
    area_r = np.maximum(reference[:, 2:]-reference[:, :2], 0).prod(axis=1)
    area_p = np.maximum(prediction[:, 2:]-prediction[:, :2], 0).prod(axis=1)
    iou = intersection / np.maximum(area_r[:, None]+area_p[None, :]-intersection, 1e-12)
    eligible = iou >= threshold
    # Maximize eligible match count first, then overlap, with one-to-one assignments.
    weight = eligible.astype(float) * (min(len(reference), len(prediction))+1) + iou*eligible
    rows, cols = linear_sum_assignment(-weight)
    pairs = [(r, c) for r, c in zip(rows, cols) if eligible[r, c]]
    return {r for r, _ in pairs}, {c for _, c in pairs}


def draw_box(ax, box, color, dashed=False):
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                           edgecolor=color, linewidth=1.0,
                           linestyle="--" if dashed else "-"))


def render_case(row, record, figures, images_root):
    reference = np.asarray(record["reference_boxes"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.7))
    image = Image.open(images_root/"test/images"/f"{row.file_name}.png").convert("RGB")
    axes[0].imshow(image)
    for box in reference:
        draw_box(axes[0], box, "#198038")
    axes[0].set_title(f"Reference: {int(row.reference_count)} cells", fontsize=12)
    for ax, arm in zip(axes[1:], ["rgb", "grayscale"]):
        view = image if arm == "rgb" else image.convert("L").convert("RGB")
        ax.imshow(view)
        pred = np.asarray(record[arm]["boxes"], dtype=float).reshape(-1, 4)
        matched_ref, matched_pred = match_boxes(reference, pred)
        for i, box in enumerate(pred):
            draw_box(ax, box, "#0072B2" if i in matched_pred else "#D55E00")
        for i, box in enumerate(reference):
            if i not in matched_ref:
                draw_box(ax, box, "#B52D8D", dashed=True)
        error = len(pred)-int(row.reference_count)
        ax.set_title(f"{arm.upper() if arm == 'rgb' else 'Grayscale'}: {len(pred)} cells (error {error:+d})", fontsize=12)
    for ax in axes:
        ax.set_xlim(0, 640)
        ax.set_ylim(640, 0)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{row.file_name} | {row.title}", fontsize=14, y=0.97)
    legend = [Patch(facecolor="none", edgecolor=c, label=t, linestyle=s) for c, t, s in [
        ("#198038", "Reference annotation", "-"),
        ("#0072B2", "Matched prediction", "-"),
        ("#D55E00", "Unmatched prediction", "-"),
        ("#B52D8D", "Unmatched reference", "--")]]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.025, "Illustrative one-to-one matching at IoU >= 0.50; unmatched boxes may reflect localization differences.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.15, wspace=0.04)
    fig.savefig(figures/f"{row.category}.png", dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Completed run containing image_alignment.csv and both arms' checkpoints")
    parser.add_argument("--images-root", type=Path, required=True,
                        help="Prepared RGB dataset containing test/images and test/labels")
    parser.add_argument("--reference-provenance", required=True,
                        help="Origin and limitations of the reference boxes used for visualization")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-boxes", action="store_true", help="Regenerate figures from saved diagnostic boxes")
    parser.add_argument("--manifest", type=Path, default=ROOT/"splits/splits.csv",
                        help="Existing shared image-level split")
    parser.add_argument("--output-root", type=Path, default=ROOT/"outputs/rgb_grayscale/diagnostics",
                        help="Destination for regenerated diagnostics; published evidence is read-only")
    args = parser.parse_args()
    output = args.output_root.resolve()
    figures = output/"figures"
    notes = (ROOT/"notes").resolve()
    if output == notes or notes in output.parents:
        raise ValueError("Choose an output directory outside published notes")
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    paired_path = RESULTS/"paired_test_counts.csv"
    manifest_path = args.manifest
    frame = pd.read_csv(paired_path).merge(
        pd.read_csv(manifest_path)[["uid", "source", "split", "count"]],
        on="uid", validate="one_to_one")
    assert len(frame) == 75 and frame.uid.is_unique and (frame.split == "test").all()
    assert (frame.reference_count == frame["count"]).all()
    for arm in ["rgb", "grayscale"]:
        assert (frame[f"{arm}_count"]-frame.reference_count == frame[f"{arm}_error"]).all()
    frame["rgb_abs"] = frame.rgb_error.abs()
    frame["gray_abs"] = frame.grayscale_error.abs()
    summaries = []
    for source, group in frame.groupby("source", sort=True):
        row = dict(source=source, n_images=len(group), mean_reference_count=group.reference_count.mean())
        for arm in ["rgb", "grayscale"]:
            error = group[f"{arm}_error"].to_numpy()
            row.update({f"{arm}_mae": float(np.abs(error).mean()),
                        f"{arm}_rmse": float(np.sqrt((error**2).mean())),
                        f"{arm}_bias": float(error.mean()),
                        f"{arm}_within_10pct": float((np.abs(error) <= .1*group.reference_count).mean()),
                        f"{arm}_sum_absolute_error": int(np.abs(error).sum())})
        row["mae_difference_grayscale_minus_rgb"] = row["grayscale_mae"]-row["rgb_mae"]
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    assert sum(summary.n_images) == 75
    for arm in ["rgb", "grayscale"]:
        assert np.isclose(np.average(summary[f"{arm}_mae"], weights=summary.n_images),
                          frame[f"{arm}_error"].abs().mean())
    summary.to_csv(output/"per_source_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x = np.arange(len(summary))
    for offset, arm, color, label in [(-.18, "rgb", "#0072B2", "RGB"), (.18, "grayscale", "#D55E00", "Grayscale")]:
        bars = ax.bar(x+offset, summary[f"{arm}_mae"], .36, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    ax.set_xticks(x, [f"{r.source}\n(n={r.n_images})" for r in summary.itertuples()], fontsize=9)
    ax.set_ylabel("Count MAE (cells/image)")
    ax.set_ylim(0, 2.5)
    ax.set_title("Counting error by acquisition source")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .015, "Descriptive analysis of the same 75 test images; all sources occur in training, validation, and test.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(figures/"mae_by_source.png", dpi=180, facecolor="white")
    plt.close(fig)
    cases = select_cases(frame)
    cases.to_csv(output/"selected_cases.csv", index=False)
    boxes_path = output/"diagnostic_boxes.json"
    verification_path = output/"verification.json"
    if args.reuse_boxes:
        saved = json.loads(boxes_path.read_text())
        verification = json.loads(verification_path.read_text())
        assert verification["paired_counts_sha256"] == sha256(paired_path)
        assert verification["manifest_sha256"] == sha256(manifest_path)
    else:
        import torch
        import ultralytics
        from ultralytics import YOLO
        assert ultralytics.__version__ == "8.4.183"
        align = pd.read_csv(args.run_root/"image_alignment.csv").set_index("uid")
        thresholds = pd.read_csv(RESULTS/"count_comparison.csv").set_index("variant").confidence.to_dict()
        image_paths, gray_inputs = [], []
        for row in frame.itertuples():
            path = args.images_root/"test/images"/f"{row.file_name}.png"
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                assert rgb.size == (640, 640)
                assert hashlib.sha256(rgb.tobytes()).hexdigest() == align.loc[row.uid, "decoded_rgb_sha256"]
                # Ultralytics ndarray sources use BGR; equal grayscale channels are order-independent.
                gray_inputs.append(np.asarray(rgb.convert("L").convert("RGB")))
            ref = reference_boxes(args.images_root/"test/labels"/f"{row.file_name}.txt")
            assert len(ref) == row.reference_count
            image_paths.append(str(path))
        saved = {row.file_name: dict(uid=row.uid,
                  reference_boxes=reference_boxes(args.images_root/"test/labels"/f"{row.file_name}.txt").tolist())
                 for i, row in enumerate(frame.itertuples()) if row.file_name in set(cases.file_name)}
        verification = dict(created_at=datetime.now(timezone.utc).isoformat(),
            purpose="Post-hoc descriptive diagnostics; no training or threshold selection",
            device=args.device, torch=torch.__version__, ultralytics=ultralytics.__version__,
            original_run="rgb_grayscale_seed0_20261026T040157Z",
            paired_counts_sha256=sha256(paired_path), manifest_sha256=sha256(manifest_path),
            reference_box_provenance=args.reference_provenance,
            decoded_image_hashes_verified=75, box_matching_iou=.5, arms={})
        for arm, inputs in [("rgb", image_paths), ("grayscale", gray_inputs)]:
            checkpoint = args.run_root/arm/"weights/best.pt"
            expected = json.loads((RESULTS/f"{arm}/training_record.json").read_text())["weights_sha256"]
            assert sha256(checkpoint) == expected
            model = YOLO(str(checkpoint))
            predictions = model.predict(inputs, conf=.01, max_det=1000, imgsz=640,
                                        batch=16, device=args.device, verbose=False)
            mismatches = []
            for row, result in zip(frame.itertuples(), predictions):
                keep = result.boxes.conf.cpu().numpy() >= thresholds[arm]
                count = int(keep.sum())
                expected_count = int(getattr(row, f"{arm}_count"))
                if count != expected_count:
                    mismatches.append(dict(file_name=row.file_name, saved=expected_count, diagnostic=count))
                if row.file_name in saved:
                    saved[row.file_name][arm] = dict(confidence_threshold=thresholds[arm],
                        boxes=result.boxes.xyxy.cpu().numpy()[keep].tolist(),
                        scores=result.boxes.conf.cpu().numpy()[keep].tolist())
            if mismatches:
                raise RuntimeError(f"Diagnostic counts differ from saved results: {arm}: {mismatches}")
            verification["arms"][arm] = dict(checkpoint_sha256=expected, confidence_threshold=thresholds[arm],
                                             matching_saved_counts=75, mismatch_count=0)
            print(f"{arm}: all 75 diagnostic counts match the saved evaluation.", flush=True)
        boxes_path.write_text(json.dumps(saved, indent=2)+"\n")
        verification_path.write_text(json.dumps(verification, indent=2)+"\n")
    matches = []
    for row in cases.itertuples():
        record = saved[row.file_name]
        render_case(row, record, figures, args.images_root)
        reference = np.asarray(record["reference_boxes"])
        for arm in ["rgb", "grayscale"]:
            pred = np.asarray(record[arm]["boxes"]).reshape(-1, 4)
            mr, mp = match_boxes(reference, pred)
            matches.append(dict(file_name=row.file_name, category=row.category, variant=arm,
                reference_count=len(reference), predicted_count=len(pred), matched_boxes=len(mr),
                unmatched_reference=len(reference)-len(mr), unmatched_prediction=len(pred)-len(mp),
                iou_threshold=.5))
    pd.DataFrame(matches).to_csv(output/"illustrative_box_matches.csv", index=False)
    print(summary[["source", "n_images", "rgb_mae", "grayscale_mae"]].to_string(index=False))
    print("Six illustrative cases and source summaries saved; original evaluation files are unchanged.")


if __name__ == "__main__":
    main()
