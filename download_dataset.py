"""
Download the Stanford 40 Actions dataset.
Source: http://vision.stanford.edu/Datasets/40actions.html

Usage:
    python download_dataset.py
"""

import os, sys, urllib.request, zipfile, shutil

URL          = "http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip"
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
ARCHIVE_PATH = os.path.join(DATA_DIR, "Stanford40_JPEGImages.zip")
IMAGES_DIR   = os.path.join(DATA_DIR, "JPEGImages")


def _progress(block, block_size, total):
    done = block * block_size
    pct  = min(100, done * 100 / total) if total > 0 else 0
    sys.stdout.write(f"\r  {pct:.1f}%  ({done/1e6:.0f} / {total/1e6:.0f} MB)")
    sys.stdout.flush()


def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.isdir(IMAGES_DIR):
        classes = [d for d in os.listdir(IMAGES_DIR)
                   if os.path.isdir(os.path.join(IMAGES_DIR, d))]
        if len(classes) >= 40:
            print(f"Dataset already present ({len(classes)} classes). Skipping download.")
            return

    print(f"Downloading Stanford 40 Actions from:\n  {URL}\n")
    try:
        urllib.request.urlretrieve(URL, ARCHIVE_PATH, reporthook=_progress)
    except Exception as e:
        print(f"\n\nDownload failed: {e}")
        print(f"\nManual option:\n  1. Download {URL}\n  2. Save to {ARCHIVE_PATH}\n  3. Re-run this script.")
        sys.exit(1)

    print(f"\nExtracting to '{DATA_DIR}' ...")
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    # Stanford40 images are provided flat; training expects class subfolders.
    moved = 0
    classes = set()
    for fname in os.listdir(IMAGES_DIR):
        fp = os.path.join(IMAGES_DIR, fname)
        if not os.path.isfile(fp) or not fname.lower().endswith(".jpg"):
            continue
        stem = os.path.splitext(fname)[0]
        if "_" not in stem:
            continue
        cls = stem.rsplit("_", 1)[0]
        cls_dir = os.path.join(IMAGES_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        shutil.move(fp, os.path.join(cls_dir, fname))
        moved += 1
        classes.add(cls)

    os.remove(ARCHIVE_PATH)
    classes = [d for d in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, d))]
    print(f"Done - {len(classes)} action classes at '{IMAGES_DIR}' (organized {moved} images).")


if __name__ == "__main__":
    download()
