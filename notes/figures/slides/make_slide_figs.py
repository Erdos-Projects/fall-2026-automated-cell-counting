"""Dark-background slide figures for the Cellpose baseline (reads cached masks, no model run)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from cellcount import cellpose_zs
from cellcount.config import get_paths
from cellcount.data import RAW_DIRNAME, load_coco
from cellcount.eda import read_rgb
from cellcount.metrics import per_image_counts
from cellcount.splits import load_splits

# Dark tokens (reference dark palette, slots 1-2; on pure black, contrast only improves).
BG, INK, INK2, GRID = "#000000", "#f0efec", "#c3c2b7", "#3a3a38"
S1, S2 = "#3987e5", "#d95926"
LABEL, PRED = "#ffd84d", "#39d5ff"  # box colours on the pink images

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 15, "axes.titlesize": 17,
    "axes.spines.top": False, "axes.spines.right": False,
})

paths = get_paths()
OUT = paths.root / "notes" / "figures" / "slides"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = paths.outputs / "cache" / "cellpose"

images, boxes = load_coco(paths.data / RAW_DIRNAME)
split = load_splits(paths.root / "splits" / "splits.csv")
truth = per_image_counts(boxes, images["uid"])
df = images[["uid", "path", "source"]].merge(split[["uid", "split"]], on="uid").merge(truth, on="uid")
df["total_true"] = df["live"] + df["dead"]
test = df[df["split"] == "test"]

DEFAULT, TUNED, MIN_AREA = (0.0, 0.4), (-4.0, 1.0), 400
masks = {k: cellpose_zs.segment(df, CACHE, cellprob_threshold=k[0], flow_threshold=k[1]) for k in (DEFAULT, TUNED)}
cells = {k: cellpose_zs.cell_areas(m) for k, m in masks.items()}

# --- Figure 1: same image, defaults vs tuned --------------------------------------------
uid = "train/142"
row = df.set_index("uid").loc[uid]
fig, axes = plt.subplots(1, 2, figsize=(13, 7.2))
for ax, (key, area, name) in zip(axes, [(DEFAULT, 0, "Default settings"), (TUNED, MIN_AREA, "Tuned on train")]):
    ax.imshow(read_rgb(row["path"]))
    for b in boxes[boxes["uid"] == uid].itertuples():
        ax.add_patch(Rectangle((b.x, b.y), b.w, b.h, fill=False, lw=1.0, ec=LABEL))
    pb = cellpose_zs.mask_boxes({uid: masks[key][uid]}, area)
    for b in pb.itertuples():
        ax.add_patch(Rectangle((b.x, b.y), b.w, b.h, fill=False, lw=1.8, ec=PRED))
    ax.set_title(f"{name}: {len(pb)} found ({row['total_true']} labelled)", color=INK, pad=10)
    ax.axis("off")
fig.legend(handles=[Line2D([], [], color=LABEL, lw=2, label="labelled cell"),
                    Line2D([], [], color=PRED, lw=2, label="found by Cellpose")],
           loc="lower center", ncol=2, frameon=False, fontsize=15, labelcolor=INK)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
fig.savefig(OUT / "slide1_defaults_vs_tuned.png", dpi=200)
print("fig1", uid, {name: n for name, n in zip(["default", "tuned"],
      [len(cellpose_zs.mask_boxes({uid: masks[k][uid]}, a)) for k, a in [(DEFAULT, 0), (TUNED, MIN_AREA)]])})

# --- Figure 2: predicted vs true on the test set ------------------------------------------
preds = {
    "Default settings": cellpose_zs.count_cells(cells[DEFAULT], test["uid"], 0),
    "Tuned on train": cellpose_zs.count_cells(cells[TUNED], test["uid"], MIN_AREA),
}
t = test.set_index("uid")["total_true"]
lim = [0, max(t.max(), max(p.max() for p in preds.values())) * 1.08]
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
for ax, (name, p) in zip(axes, preds.items()):
    p = p.reindex(t.index)
    mae = (p - t).abs().mean()
    ax.plot(lim, lim, color=INK2, lw=1.2, ls="--", zorder=1)
    ax.text(lim[1] * 0.03, lim[1] * 0.36, "perfect count", color=INK2, fontsize=12, ha="left")
    ax.scatter(t, p, s=70, color=S1, edgecolor=BG, linewidth=1.5, zorder=2)
    ax.set(xlim=lim, ylim=lim, xlabel="true count", aspect="equal")
    ax.set_title(f"{name}\nMAE {mae:.1f} cells / image", color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("predicted count")
fig.tight_layout()
fig.savefig(OUT / "slide2_pred_vs_true.png", dpi=200)
print("fig2", {k: round(float((v.reindex(t.index) - t).abs().mean()), 2) for k, v in preds.items()})

# --- Figure 3: MAE, random split vs unseen session ---------------------------------------
res = pd.read_csv(paths.outputs / "results" / "baseline_cellpose.csv")
methods = {"B0 session mean": "Session-average\nshortcut", "B2 Cellpose-SAM (tuned on train)": "Cellpose-SAM\n(tuned)"}
levels = {"random split": ("Seen sessions (test split)", S1), "leave-one-session-out": ("Unseen session", S2)}
fig, ax = plt.subplots(figsize=(10, 6))
x, w = np.arange(len(methods)), 0.36
for i, (lvl, (lab, col)) in enumerate(levels.items()):
    vals = []
    for m in methods:
        r = res[(res["method"] == m) & (res["level"] == lvl)]
        r = r[r["fold"].isin(["test", "pooled"])]
        vals.append(float(r["mae_total"].iloc[0]))
    xs = x + (i - 0.5) * (w + 0.02)
    ax.bar(xs, vals, width=w, color=col, label=lab)
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", va="bottom", color=INK, fontsize=16, fontweight="bold")
    print("fig3", lvl, vals)
ax.set_xticks(x, list(methods.values()), color=INK, fontsize=16)
ax.set_ylabel("count error (MAE, cells / image)")
ax.set_ylim(0, 17)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="x", length=0)
ax.legend(frameon=False, labelcolor=INK, loc="upper right", fontsize=14)
ax.set_title("Lower is better", color=INK2, loc="left", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "slide2_mae_unseen_session.png", dpi=200)
print("saved to", OUT)
