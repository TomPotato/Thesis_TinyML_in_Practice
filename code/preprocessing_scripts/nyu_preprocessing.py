"""
Convert NYU Depth V2 (labeled subset) into the folder layout expected by
train_tinyresnet.py, using the depth maps to auto-label each image:

    data/train/{clear,left_blocked,center_blocked,right_blocked}/*.png
    data/val/{clear,left_blocked,center_blocked,right_blocked}/*.png

How the labeling works
  1. Look only at the lower part of the frame (ceilings don't block a rover).
  2. Split that band into left / center / right thirds.
  3. In each third take a low percentile of the depth (robust to noisy pixels).
  4. A third is "blocked" if that distance is below --threshold meters.
  5. Map the blocked/not-blocked pattern to one of the 4 classes.
     Left+right blocked with a clear center is ambiguous and is skipped.

Download the labeled subset (about 2.8 GB):
    nyu_depth_v2_labeled.mat   (from the NYU Depth V2 dataset page)

Requirements:
    pip install h5py opencv-python numpy

Usage:
    python nyu_to_dataset.py --mat nyu_depth_v2_labeled.mat --out_dir ./data
"""

import argparse
import os
import random
from collections import Counter

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


def zone_distances(depth, band_top=0.4):
    """Nearest-ish distance (5th percentile) in left/center/right thirds of
    the lower part of the frame. Returns [left, center, right] in meters."""
    h, _ = depth.shape
    band = depth[int(h * band_top):, :]
    out = []
    for zone in np.array_split(band, 3, axis=1):
        valid = zone[zone > 0.1]  # drop invalid zero readings
        out.append(float(np.percentile(valid, 5)) if valid.size else 99.0)
    return out


def make_label(left, center, right, threshold):
    """Map three zone distances to a class name, or None if ambiguous."""
    blocked_l, blocked_c, blocked_r = left < threshold, center < threshold, right < threshold
    if not (blocked_l or blocked_c or blocked_r):
        return "clear"
    if blocked_c:
        return "center_blocked"
    if blocked_l and not blocked_r:
        return "left_blocked"
    if blocked_r and not blocked_l:
        return "right_blocked"
    return None  # left + right blocked, center clear: ambiguous, skip


def preprocess(rgb):
    """RGB -> 96x96 grayscale, matching the ESP-EYE input. The whole frame is
    squashed (not cropped) so left/center/right zones stay meaningful."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def load_nyu(mat_path):
    import h5py  # imported here so the helper functions above are testable alone
    f = h5py.File(mat_path, "r")
    return f["images"], f["depths"]


def get_depth(depths, i):
    return np.transpose(depths[i], (1, 0))


def get_frame(images, depths, i):
    # HDF5 arrays are stored transposed: images (N,3,W,H), depths (N,W,H)
    rgb = np.transpose(images[i], (2, 1, 0))
    depth = np.transpose(depths[i], (1, 0))
    return rgb, depth


def save_image(img, out_dir, subset, label, name):
    folder = os.path.join(out_dir, subset, label)
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, name), img)


def make_preview(out_dir, per_class=8):
    """Contact sheet of random training samples per class for a quick visual
    sanity check of the automatic labels."""
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
        print(f"Preview sheet written to {path}  (check that labels look right!)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True, help="Path to nyu_depth_v2_labeled.mat")
    ap.add_argument("--out_dir", default="./data")
    ap.add_argument("--threshold", type=float, default=1.0, help="Blocked if closer than this (m)")
    ap.add_argument("--band_top", type=float, default=0.4, help="Ignore the top fraction of the frame")
    ap.add_argument("--val_fraction", type=float, default=0.2, help="Last fraction of the file goes to val")
    ap.add_argument("--max_per_class", type=int, default=0, help="Cap per class per subset (0 = no cap)")
    ap.add_argument("--add_flips", action="store_true",
                    help="Also save mirrored train images with left/right labels swapped")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    print(f"Opening {args.mat} ...", flush=True)
    images, depths = load_nyu(args.mat)
    n = images.shape[0]
    split_at = int(n * (1 - args.val_fraction))
    print(f"{n} frames; first {split_at} -> train, last {n - split_at} -> val (split by file position, not random)")

    # Pass 1: label everything, cheaply (depth only)
    items = []  # (index, label, subset)
    skipped = 0
    for i in range(n):
        if i % 100 == 0:
            print(f"  labeling frame {i}/{n} ...", flush=True)
        depth = get_depth(depths, i)
        label = make_label(*zone_distances(depth, args.band_top), args.threshold)
        if label is None:
            skipped += 1
            continue
        items.append((i, label, "train" if i < split_at else "val"))
    print(f"Skipped {skipped} ambiguous frames; {len(items)} frames will be written", flush=True)
    if not items:
        print("Nothing to write. Try a different --threshold (e.g. 1.5) or check that the .mat file is the labeled subset.")
        return

    # Optional per-class cap, applied separately to train and val
    if args.max_per_class > 0:
        kept = []
        for subset in ("train", "val"):
            for cls in CLASSES:
                group = [it for it in items if it[1] == cls and it[2] == subset]
                random.shuffle(group)
                kept.extend(group[:args.max_per_class])
        items = kept

    # Pass 2: write images
    counts = Counter()
    for k, (i, label, subset) in enumerate(items):
        if k % 100 == 0:
            print(f"  writing image {k}/{len(items)} ...", flush=True)
        rgb, _ = get_frame(images, depths, i)
        small = preprocess(rgb)
        save_image(small, args.out_dir, subset, label, f"nyu_{i:05d}.png")
        counts[(subset, label)] += 1
        if args.add_flips and subset == "train":
            save_image(cv2.flip(small, 1), args.out_dir, subset, FLIP_SWAP[label], f"nyu_{i:05d}_flip.png")
            counts[(subset, FLIP_SWAP[label])] += 1

    print("\nImages written:")
    for subset in ("train", "val"):
        print(f"  {subset}: " + ", ".join(f"{c}={counts[(subset, c)]}" for c in CLASSES))
    print("\nIf one class dominates, use --max_per_class or class weights when training.")

    make_preview(args.out_dir)


if __name__ == "__main__":
    main()