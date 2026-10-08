# Dataset Download Links

## Roboflow dataset

[Roboflow Cell Counting](https://universe.roboflow.com/cell-counting/cell-counting-zeqo5) | **Primary, in use** | 500 unique images (the public version's ~1,200 includes 3× augmented training copies)

Our fork: workspace `dingshan-deng`, project ID `cell-counting-zeqo5-7svoy` (display name "cell counting"), raw COCO export. The fork has no generated version yet, so the API route (`download_roboflow`) only works after someone generates one.

> **Never paste API keys or signed download links into this repo.**
> - Local: put `ROBOFLOW_EXPORT_URL=...` (and optionally `ROBOFLOW_API_KEY=...`) in `.env` (gitignored, template in `.env.example`).
> - Colab: add the same names under Secrets (key icon in the left sidebar).

Download with the repo utils (works locally and on Colab), or just run `notebooks/00_setup_and_download.ipynb`:

```python
from cellcount.config import get_paths, get_secret
from cellcount.data import download_export

download_export(get_secret("ROBOFLOW_EXPORT_URL"), get_paths().data)
```

### Why the public version shows ~1,200 images (verified 2026-10-06)

The Universe page's download snippet (`rf.workspace("cell-counting").project("cell-counting-zeqo5").version(1).download("coco")`) fetches the public version 1, not a larger dataset. We downloaded it with `download_roboflow(..., version=1, workspace="cell-counting", project="cell-counting-zeqo5")` and compared it with our raw export by original file name:

| | Public v1 | Our raw export |
|---|---|---|
| Images | 1,200 (train 1,050 / valid 100 / test 50) | 500 (350 / 100 / 50) |
| Unique source images | 500 | 500 |
| Shared with the other set | all 500 | all 500 |
| Sources | same 5 `2022_12_7_*` sessions, 100 each | same |
| Boxes per image | identical to the raw original in every copy | |

Its `README.roboflow.txt` explains the difference: every image was stretched to 640×640, and each training image has 3 augmented versions (salt-and-pepper noise on 5% of pixels). So the 1,050 training images are 350 originals × 3, and valid and test are unchanged.

**Don't use it.** It adds no new cells, sessions, or counts, and because `splits/splits.csv` reassigns the 500 originals, its copies would leak validation and test images into training. If we want noise augmentation, we apply it ourselves at training time. More data has to come from other datasets (see `plan.md` §5.3).

Note: the `universe.roboflow.com/ds/...?key=...` raw URL returns HTTP 403 (Cloudflare check) to scripts. Only `app.roboflow.com/ds/...` export links from our own workspace work with `download_export()`.
