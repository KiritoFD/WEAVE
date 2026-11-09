#!/usr/bin/env bash
set -e

WORK_DIR="/root/Workspace/data/wikiart"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

KAGGLE_BIN="/opt/conda/envs/cu121/bin/kaggle"
LOG_FILE="$WORK_DIR/download_progress.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== [$(date)] Starting WikiArt Kaggle Dataset Download ==="

# 1. Download and extract test.zip (~11.89 GB, 23,817 images)
if [ ! -d "$WORK_DIR/test" ] || [ $(find "$WORK_DIR/test" -name "*.jpg" | wc -l) -lt 20000 ]; then
    echo "[$(date)] Downloading test.zip..."
    $KAGGLE_BIN competitions download -c painter-by-numbers -f test.zip
    
    echo "[$(date)] Unpacking test.zip..."
    mkdir -p "$WORK_DIR/test"
    # Note: Kaggle may download as test.zip or test.zip.zip
    ZIP_NAME="test.zip"
    if [ ! -f "$ZIP_NAME" ] && [ -f "test.zip.zip" ]; then
        ZIP_NAME="test.zip.zip"
    fi
    unzip -q -o "$ZIP_NAME" -d "$WORK_DIR/test"
    echo "[$(date)] test.zip extracted. Current test images count: $(find "$WORK_DIR/test" -name '*.jpg' | wc -l)"
    rm -f "$ZIP_NAME"
else
    echo "[$(date)] test/ already exists and complete ($(find "$WORK_DIR/test" -name '*.jpg' | wc -l) images). Skipping."
fi

# 2. Download and extract train.zip (~38.69 GB, 79,433 images)
if [ ! -d "$WORK_DIR/train" ] || [ $(find "$WORK_DIR/train" -name "*.jpg" | wc -l) -lt 70000 ]; then
    echo "[$(date)] Downloading train.zip..."
    $KAGGLE_BIN competitions download -c painter-by-numbers -f train.zip
    
    echo "[$(date)] Unpacking train.zip..."
    mkdir -p "$WORK_DIR/train"
    ZIP_NAME="train.zip"
    if [ ! -f "$ZIP_NAME" ] && [ -f "train.zip.zip" ]; then
        ZIP_NAME="train.zip.zip"
    fi
    unzip -q -o "$ZIP_NAME" -d "$WORK_DIR/train"
    echo "[$(date)] train.zip extracted. Current train images count: $(find "$WORK_DIR/train" -name '*.jpg' | wc -l)"
    rm -f "$ZIP_NAME"
else
    echo "[$(date)] train/ already exists and complete ($(find "$WORK_DIR/train" -name '*.jpg' | wc -l) images). Skipping."
fi

echo "=== [$(date)] Download and Extraction Finished Successfully! ==="
echo "Total WikiArt images ready for VAE training: $(find "$WORK_DIR" -name '*.jpg' | wc -l)"
