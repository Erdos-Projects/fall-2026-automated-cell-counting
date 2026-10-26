"""Align a Roboflow YOLO export to the team's frozen image-level manifest."""
from pathlib import Path
import hashlib
import json
import math
import re
import pandas as pd
from PIL import Image


def original_image_key(name):
    return re.sub(r"_(?:jpe?g|png|tiff?)\.rf\..*$", "", name, flags=re.IGNORECASE)


def yolo_label_path(image):
    parts = list(Path(image).parts)
    if "images" not in parts:
        raise ValueError(f"Expected an images directory: {image}")
    parts[len(parts)-1-parts[::-1].index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def prepare_frozen_yolo(export_root, manifest_path, output_root):
    export_root, manifest_path, output_root = map(Path, (export_root, manifest_path, output_root))
    manifest = pd.read_csv(manifest_path)
    if not {"uid", "file_name", "count", "split"}.issubset(manifest.columns):
        raise ValueError("The frozen manifest is missing required columns")
    if len(manifest) != 500 or manifest.split.value_counts().to_dict() != {"train":351,"test":75,"valid":74}:
        raise ValueError("Expected the frozen 351/74/75 split")
    manifest["original_key"] = manifest.file_name.map(original_image_key)
    if not manifest.uid.is_unique or not manifest.original_key.is_unique:
        raise ValueError("Ambiguous identities in the frozen manifest")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(p for p in export_root.rglob("*") if p.is_file() and p.suffix.lower() in extensions and p.parent.name == "images")
    lookup = {}
    for p in images:
        key = original_image_key(p.name)
        if key in lookup:
            raise ValueError("Duplicate originals or augmented copies found; expected 500 unique originals")
        lookup[key] = p
    if len(images) != 500 or set(lookup) != set(manifest.original_key):
        raise ValueError("Downloaded images do not match the team's 500 originals")
    checked = []
    for row in manifest.itertuples():
        image = lookup[row.original_key]
        label = yolo_label_path(image)
        if not label.is_file():
            raise FileNotFoundError(label)
        lines = [s.split() for s in label.read_text().splitlines() if s.strip()]
        if len(lines) != row.count:
            raise ValueError(f"Count mismatch for {row.original_key}: {len(lines)} vs {row.count}")
        normalized = []
        for fields in lines:
            if len(fields) != 5:
                raise ValueError(f"Expected detection box labels, not polygons: {label}")
            category, cx, cy, w, h = map(float, fields)
            if not all(math.isfinite(v) for v in (category,cx,cy,w,h)) or not category.is_integer() or category < 0:
                raise ValueError(f"Invalid class or coordinate in {label}")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f"Invalid normalized box in {label}")
            normalized.append(f"0 {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}")
        with Image.open(image) as im:
            if im.size != (640,640):
                raise ValueError(f"Expected 640x640 source images, found {im.size}: {image}")
        checked.append((row,image,normalized))
    marker = output_root / "alignment_metadata.json"
    signature = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if output_root.exists() and any(output_root.iterdir()) and not marker.exists():
        raise FileExistsError("Output contains an unfinished/unknown dataset. Preserve it and choose a new directory.")
    if marker.exists() and json.loads(marker.read_text())["manifest_sha256"] != signature:
        raise ValueError("Existing output uses a different split")
    inventory = []
    for row,image,lines in checked:
        images_dir = output_root / row.split / "images"
        labels_dir = output_root / row.split / "labels"
        images_dir.mkdir(parents=True,exist_ok=True)
        labels_dir.mkdir(parents=True,exist_ok=True)
        stem = row.uid.replace("/","_")
        target = images_dir / f"{stem}.png"
        with Image.open(image) as im:
            rgb = im.convert("RGB")
            rgb.save(target)
            pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        (labels_dir/f"{stem}.txt").write_text("\n".join(lines))
        inventory.append(dict(uid=row.uid,original_key=row.original_key,source_file=str(image),
                              split=row.split,count=row.count,path=str(target.resolve()),
                              decoded_rgb_sha256=pixel_hash))
    config = output_root / "data.yaml"
    config.write_text(f"path: {output_root.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: Cell\n")
    pd.DataFrame(inventory).to_csv(output_root/"image_alignment.csv",index=False)
    meta = dict(manifest_sha256=signature,n_images=500,n_boxes=sum(r.count for r,_,_ in checked),
                split_counts=manifest.split.value_counts().to_dict(),classes=["Cell"],
                identity_check="Unique original filename plus exact annotation count per image",
                geometry_caveat="No pixel/coordinate equality claim against an unavailable team raw export")
    marker.write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(json.dumps(meta,ensure_ascii=False,indent=2))
    return config
