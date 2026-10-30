"""
Convert COCO object annotations into the folder layout expected by
train_tinyresnet.py, labeling each image from its bounding boxes:

    data/train/{clear,left_blocked,center_blocked,right_blocked}/*.png
    data/val/{clear,left_blocked,center_blocked,right_blocked}/*.png

How the labeling works
  * Only obstacle-like categories count (chair, couch, suitcase, ...).
  * A box counts as an obstacle only if it is big enough (--min_area, fraction
    of the image) AND reaches the lower part of the frame (--min_bottom).
    Objects whose base is low in the image are close to the camera and on the
    ground, which is a rough stand-in for "in the rover's way".
  * A box blocks a left/center/right third if it covers at least
    --min_zone_overlap of that third's width.
  * Same blocked-pattern -> class mapping as nyu_to_dataset.py.
  * "clear" = no obstacle-class boxes AND no other large object either, so
    unannotated big things don't sneak into the clear class.

This is noisier than depth-based labeling (box size is only a proxy for
distance), so spot-check the preview sheet this script writes.

Requirements:
    pip install opencv-python numpy

Usage:
    python coco_to_retina.py \
        --ann annotations/instances_val2017.json \
        --images val2017 \
        --out_dir ./data_coco \
        --max_per_class 800 --add_flips if you want to mirror left/right images for training and increase the dataset size
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from builtins import *

import cv2
import numpy as np

CLASSES = ["clear", "left_blocked", "center_blocked", "right_blocked"]
FLIP_SWAP = {
    "clear": "clear",
    "center_blocked": "center_blocked",
    "left_blocked": "right_blocked",
    "right_blocked": "left_blocked",
}
IMG_SIZE = 96

OBSTACLE_CATEGORIES = {
    "chair", "couch", "bench", "suitcase", "potted plant", "bed",
    "dining table", "backpack", "bicycle", "refrigerator",
    "toilet"
}


def box_zone_overlaps(x, w_box, img_w):
    """Fraction of each third's width covered by the box, as [left, center, right]."""
    zone_w = img_w / 3.0
    overlaps = []
    for k in range(3):
        z0, z1 = k * zone_w, (k + 1) * zone_w
        inter = max(0.0, min(x + w_box, z1) - max(x, z0))
        overlaps.append(inter / zone_w)
    return overlaps


def blocked_zones(boxes, img_w, img_h, min_area, min_bottom, min_overlap):
    """Return [left, center, right] booleans from a list of (x, y, w, h) boxes."""
    blocked = [False, False, False]
    for (x, y, w, h) in boxes:
        if (w * h) / (img_w * img_h) < min_area:
            continue
        if (y + h) / img_h < min_bottom:
            continue
        for k, frac in enumerate(box_zone_overlaps(x, w, img_w)):
            if frac >= min_overlap:
                blocked[k] = True
    return blocked


def label_from_blocked(blocked):
    left, center, right = blocked
    if not (left or center or right):
        return "clear"
    if center:
        return "center_blocked"
    if left and not right:
        return "left_blocked"
    if right and not left:
        return "right_blocked"
    return None  # left + right with clear center: ambiguous, skip


def preprocess_gray(gray):
    return cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def save_image(img, out_dir, subset, label, name):
    folder = os.path.join(out_dir, subset, label)
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, name), img)


def make_preview(out_dir, per_class=8):
    rows = []
    for cls in CLASSES:
        folder = os.path.join(out_dir, "train", cls)
        if not os.path.isdir(folder):
            continue
        files = sorted(os.listdir(folder))
        random.shuffle(files)
        imgs = [cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE) for f in files[:per_class]]
        imgs += [np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)] * (per_class - len(imgs))
        row = cv2.hconcat(imgs)
        cv2.putText(row, cls, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        rows.append(row)
    if rows:
        path = os.path.join(out_dir, "preview.png")
        cv2.imwrite(path, cv2.vconcat(rows))
        print(f"Preview sheet written to {path}  (check that images correspond!)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="COCO instances_*.json")
    ap.add_argument("--images", required=True, help="Folder with the matching COCO images")
    ap.add_argument("--out_dir", default="./data_coco")
    ap.add_argument("--min_area", type=float, default=0.04, help="Min box area as fraction of image")
    ap.add_argument("--min_bottom", type=float, default=0.55, help="Box bottom edge must be below this fraction of image height")
    ap.add_argument("--min_zone_overlap", type=float, default=0.25, help="Box must cover this fraction of a third's width")
    ap.add_argument("--clear_max_area", type=float, default=0.15, help="Any object bigger than this disqualifies a 'clear' image")
    ap.add_argument("--include_person", action="store_true", help="Treat people as obstacles too")
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--max_per_class", type=int, default=0, help="Cap per class per subset (0 = no cap)")
    ap.add_argument("--add_flips", action="store_true",
                    help="Also save mirrored train images with left/right labels swapped")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.ann) as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    obstacle_names = set(OBSTACLE_CATEGORIES)
    if args.include_person:
        obstacle_names.add("person")

    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):
            continue
        anns_by_image[a["image_id"]].append(a)

    # Pass 1: label every image from its boxes
    items = []  # (image_info, label, subset)
    skipped = 0
    for img in coco["images"]:
        iw, ih = img["width"], img["height"]
        anns = anns_by_image.get(img["id"], [])
        obstacle_boxes = [a["bbox"] for a in anns if cat_names[a["category_id"]] in obstacle_names]

        blocked = blocked_zones(obstacle_boxes, iw, ih, args.min_area, args.min_bottom, args.min_zone_overlap)
        label = label_from_blocked(blocked)

        if label == "clear":
            # Require a genuinely empty-looking scene: no listed obstacle boxes
            # at all, and no other large object either.
            has_any_obstacle = len(obstacle_boxes) > 0
            has_big_object = any((a["bbox"][2] * a["bbox"][3]) / (iw * ih) > args.clear_max_area for a in anns)
            if has_any_obstacle or has_big_object:
                label = None

        if label is None:
            skipped += 1
            continue

        # Deterministic split by image id so reruns never mix train/val
        subset = "val" if (img["id"] % 100) < int(args.val_fraction * 100) else "train"
        items.append((img, label, subset))

    print(f"{len(items)} labeled images, {skipped} skipped (not suitable)")

    # Optional per-class cap
    if args.max_per_class > 0:
        kept = []
        for subset in ("train", "val"):
            for cls in CLASSES:
                group = [it for it in items if it[1] == cls and it[2] == subset]
                random.shuffle(group)
                kept.extend(group[:args.max_per_class])
        items = kept

    # Pass 2: load, convert, write
    counts = Counter()
    for img, label, subset in items:
        path = os.path.join(args.images, img["file_name"])
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  missing image, skipping: {path}")
            continue
        small = preprocess_gray(gray)
        name = f"coco_{img['id']:012d}.png"
        save_image(small, args.out_dir, subset, label, name)
        counts[(subset, label)] += 1
        if args.add_flips and subset == "train":
            save_image(cv2.flip(small, 1), args.out_dir, subset, FLIP_SWAP[label], name.replace(".png", "_flip.png"))
            counts[(subset, FLIP_SWAP[label])] += 1

    print("\nImages written:")
    for subset in ("train", "val"):
        print(f"  {subset}: " + ", ".join(f"{c}={counts[(subset, c)]}" for c in CLASSES))
    print("\nIf one class dominates, use --max_per_class or class weights when training.")

    make_preview(args.out_dir)


if __name__ == "__main__":
    main()