"""Zero-shot Cellpose-SAM baseline (models B1 and B2: default and tuned settings): segment every cell with the published model, then count.

The network is never trained on our images. Three numbers are chosen on **training images only**
(``tune``): the two mask thresholds Cellpose exposes (``cellprob_threshold``, ``flow_threshold``)
and a minimum cell area that drops debris and fragments. Each choice minimizes total-count MAE.

The network's outputs (flows and cell probability) don't depend on the thresholds, so ``segment``
caches them once per image and builds masks for any threshold setting from that cache.

Live/Dead is a later extension (``masks_to_cells``, ``classify``, ``fit_rule``): Cellpose knows
nothing about viability, but trypan blue stains dead cells, so a cell can be called Dead when its
blueness ``B - (R + G) / 2``, measured relative to the image background, passes a cut-off.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from tqdm.auto import tqdm

from .eda import read_rgb


# Chosen by ``tune`` on the training images (notebooks_local/02_baseline_cellpose.ipynb). Cellpose's own
# defaults (0, 0.4) find only a fraction of the pale live cells.
EVAL_KWARGS = {"cellprob_threshold": -6.0, "flow_threshold": 1.0}
THRESHOLD_GRID = {"cellprob_threshold": (0.0, -2.0, -4.0, -6.0, -8.0), "flow_threshold": (0.4, 0.7, 1.0)}
AREA_GRID = (0, 100, 200, 300, 400, 500, 600)


def _device():
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def segment(images: pd.DataFrame, cache_dir: Path, model: str = "cpsam_v2", **eval_kwargs) -> dict[str, np.ndarray]:
    """Cellpose label masks per image uid, built from cached network outputs.

    ``eval_kwargs`` (default ``EVAL_KWARGS``) are the mask thresholds. Network outputs are cached
    in ``<cache_dir>/<model>/flows/`` and masks in a folder named after the thresholds, so re-runs
    skip the model and masks from different settings never mix.
    """
    eval_kwargs = eval_kwargs or EVAL_KWARGS
    settings = "_".join(f"{k}={float(v)}" for k, v in sorted(eval_kwargs.items()))
    flow_dir = Path(cache_dir) / model / "flows"
    mask_dir = Path(cache_dir) / model / settings
    flow_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    masks, net, device = {}, None, None
    for uid, path in tqdm(zip(images["uid"], images["path"]), total=len(images), desc=f"cellpose {settings}"):
        name = f"{uid.replace('/', '_')}.npz"
        if (mask_dir / name).exists():
            masks[uid] = np.load(mask_dir / name)["mask"]
            continue
        from cellpose import dynamics

        device = device or _device()
        if (flow_dir / name).exists():
            f = np.load(flow_dir / name)
            dP, cellprob = f["dP"].astype(np.float32), f["cellprob"].astype(np.float32)
        else:
            if net is None:
                from cellpose import models

                net = models.CellposeModel(gpu=device.type != "cpu", device=device, pretrained_model=model)
            _, (_, dP, cellprob), _ = net.eval(read_rgb(path), compute_masks=False)
            np.savez_compressed(flow_dir / name, dP=dP.astype(np.float16), cellprob=cellprob.astype(np.float16))
            dP, cellprob = dP.astype(np.float16).astype(np.float32), cellprob.astype(np.float16).astype(np.float32)
        # Same post-processing as CellposeModel.eval for a 2D image at its own size.
        mask = dynamics.resize_and_compute_masks(dP, cellprob, niter=200, device=device, **eval_kwargs)
        mask = mask.astype(np.uint16)
        np.savez_compressed(mask_dir / name, mask=mask)
        masks[uid] = mask
    return masks


def cell_areas(masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per segmented cell: ``uid, area`` (px)."""
    frames = [pd.DataFrame({"uid": uid, "area": np.bincount(m.ravel())[1:]}) for uid, m in masks.items()]
    cells = pd.concat(frames, ignore_index=True)
    return cells[cells["area"] > 0]


def count_cells(cells: pd.DataFrame, uids: pd.Series, min_area: float = 0) -> pd.Series:
    """Cells of at least ``min_area`` px per image (0 for images with none), indexed by uid."""
    kept = cells[cells["area"] >= min_area]
    return kept.groupby("uid").size().reindex(pd.Index(uids, name="uid"), fill_value=0)


def fit_min_area(cells: pd.DataFrame, truth: pd.DataFrame, area_grid: tuple[float, ...] = AREA_GRID) -> tuple[float, float]:
    """``(min_area, mae)``: the area cut-off minimizing total-count MAE on the images in ``truth``
    (``uid, total_true``). Use training images only."""
    t = truth.set_index("uid")["total_true"]
    maes = {a: (count_cells(cells, t.index, a) - t).abs().mean() for a in area_grid}
    best = min(maes, key=maes.get)
    return float(best), float(maes[best])


def tune(cells_by_setting: dict[tuple[float, float], pd.DataFrame], truth: pd.DataFrame,
         area_grid: tuple[float, ...] = AREA_GRID) -> pd.DataFrame:
    """Fit the minimum area for every ``(cellprob_threshold, flow_threshold)`` setting on ``truth``.

    Returns one row per setting (``cellprob_threshold, flow_threshold, min_area, mae``), best first.
    """
    rows = []
    for (cp, fl), cells in cells_by_setting.items():
        min_area, mae = fit_min_area(cells, truth, area_grid)
        rows.append({"cellprob_threshold": cp, "flow_threshold": fl, "min_area": min_area, "mae": mae})
    return pd.DataFrame(rows).sort_values("mae", ignore_index=True)


def threshold_grid(grid: dict[str, tuple[float, ...]] = THRESHOLD_GRID) -> list[tuple[float, float]]:
    """Every ``(cellprob_threshold, flow_threshold)`` pair in ``grid``."""
    return list(product(grid["cellprob_threshold"], grid["flow_threshold"]))


def mask_boxes(masks: dict[str, np.ndarray], min_area: float = 0) -> pd.DataFrame:
    """Bounding boxes in the ``load_coco`` schema (``uid, category, x, y, w, h``, plus ``area``),
    one per cell of at least ``min_area`` px, all with category ``Cell``."""
    frames = []
    for uid, mask in masks.items():
        if mask.max() == 0:
            continue
        p = regionprops_table(mask, properties=("bbox", "area"))
        frames.append(pd.DataFrame({
            "uid": uid, "category": "Cell",
            "x": p["bbox-1"], "y": p["bbox-0"], "w": p["bbox-3"] - p["bbox-1"], "h": p["bbox-2"] - p["bbox-0"],
            "area": p["area"],
        }))
    boxes = pd.concat(frames, ignore_index=True)
    return boxes[boxes["area"] >= min_area].reset_index(drop=True)


# --- Live/Dead extension (not part of the current total-count baseline) ---


def masks_to_cells(masks: dict[str, np.ndarray], images: pd.DataFrame) -> pd.DataFrame:
    """One row per segmented cell: box (``x, y, w, h``), ``area``, mean colour, and ``blue_rel``,
    the cell's blueness minus that of its image's background (median of the unsegmented pixels)."""
    paths = images.set_index("uid")["path"]
    frames = []
    for uid, mask in masks.items():
        if mask.max() == 0:
            continue
        rgb = read_rgb(paths[uid]).astype(float)
        props = regionprops_table(mask, intensity_image=rgb, properties=("bbox", "area", "intensity_mean"))
        bg = np.median(rgb[mask == 0], axis=0)
        cells = pd.DataFrame(
            {
                "uid": uid,
                "x": props["bbox-1"], "y": props["bbox-0"],
                "w": props["bbox-3"] - props["bbox-1"], "h": props["bbox-2"] - props["bbox-0"],
                "area": props["area"],
                "r": props["intensity_mean-0"], "g": props["intensity_mean-1"], "b": props["intensity_mean-2"],
                "bg_blue": bg[2] - (bg[0] + bg[1]) / 2,
            }
        )
        frames.append(cells)
    cells = pd.concat(frames, ignore_index=True)
    cells["blue_rel"] = cells["b"] - (cells["r"] + cells["g"]) / 2 - cells["bg_blue"]
    return cells


def classify(cells: pd.DataFrame, dead_threshold: float, min_area: float = 0) -> pd.DataFrame:
    """Keep cells of at least ``min_area`` px and label them Dead above the blueness threshold."""
    kept = cells[cells["area"] >= min_area]
    return kept.assign(category=np.where(kept["blue_rel"] > dead_threshold, "Dead", "Live"), conf=1.0)


def fit_rule(cells: pd.DataFrame, truth: pd.DataFrame, area_grid: tuple[float, ...] = AREA_GRID) -> tuple[float, float]:
    """``(min_area, dead_threshold)`` fitted on the images in ``truth`` (``uid, live_true, dead_true``).

    The minimum area minimizes total-count MAE; the threshold then minimizes Dead-count MAE.
    Use training images only.
    """
    t = truth.set_index("uid")
    cells = cells[cells["uid"].isin(t.index)]

    def mae(counts: pd.Series, target: pd.Series) -> float:
        return (counts.reindex(t.index, fill_value=0) - target).abs().mean()

    total = t["live_true"] + t["dead_true"]
    min_area = min(area_grid, key=lambda a: mae(cells[cells["area"] >= a].groupby("uid").size(), total))
    kept = cells[cells["area"] >= min_area]
    candidates = np.unique(np.quantile(kept["blue_rel"], np.linspace(0.8, 0.999, 200)))
    dead_threshold = min(candidates, key=lambda c: mae((kept["blue_rel"] > c).groupby(kept["uid"]).sum(), t["dead_true"]))
    return float(min_area), float(dead_threshold)
