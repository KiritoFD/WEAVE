"""Robust Resumable Downloader for Kaggle Competition Files.
Automatically handles 302 presigned URLs, HTTP Range resuming, chunked streaming,
and retry on network disconnects.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
import requests

def download_file_resumable(
    competition: str,
    file_name: str,
    dest_dir: Path,
    username: str,
    key: str,
    max_retries: int = 50,
    chunk_size: int = 4 * 1024 * 1024, # 4MB chunks
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / file_name
    part_path = dest_dir / f"{file_name}.part"

    api_url = f"https://www.kaggle.com/api/v1/competitions/data/download/{competition}/{file_name}"
    auth = (username, key)

    for attempt in range(1, max_retries + 1):
        try:
            # 1. Get fresh 302 presigned URL
            r_head = requests.get(api_url, auth=auth, allow_redirects=False, timeout=15)
            if r_head.status_code != 302:
                print(f"[Attempt {attempt}] Failed to get redirect URL: status {r_head.status_code}, response: {r_head.text[:100]}")
                time.sleep(3)
                continue

            direct_url = r_head.headers.get("Location")
            if not direct_url:
                print(f"[Attempt {attempt}] No Location header in redirect response")
                time.sleep(3)
                continue

            # 2. Check existing downloaded bytes for resumption
            downloaded = part_path.stat().st_size if part_path.exists() else 0
            headers = {}
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
                print(f"[{file_name}] Resuming from byte offset {downloaded} ({downloaded / (1024**2):.1f} MB)...", flush=True)
            else:
                print(f"[{file_name}] Starting fresh download...", flush=True)

            # 3. Stream download from direct URL
            with requests.get(direct_url, headers=headers, stream=True, timeout=30) as r_stream:
                if r_stream.status_code not in (200, 206):
                    print(f"[Attempt {attempt}] Storage returned status {r_stream.status_code}")
                    if r_stream.status_code == 416: # Range not satisfiable (maybe already complete)
                        print(f"[{file_name}] Range 416: file might be fully downloaded.")
                        break
                    time.sleep(3)
                    continue

                total_size = int(r_stream.headers.get("Content-Length", 0)) + downloaded
                mode = "ab" if downloaded > 0 else "wb"

                t0 = time.time()
                bytes_this_attempt = 0

                with open(part_path, mode) as f:
                    for chunk in r_stream.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            bytes_this_attempt += len(chunk)
                            speed_mb = (bytes_this_attempt / (1024**2)) / max(time.time() - t0, 1e-4)
                            pct = (downloaded / total_size * 100.0) if total_size > 0 else 0.0
                            print(
                                f"\r[{file_name}] {downloaded / (1024**3):.2f}/{total_size / (1024**3):.2f} GB "
                                f"({pct:5.1f}%) | Speed: {speed_mb:6.1f} MB/s",
                                end="", flush=True
                            )

            print(f"\n[{file_name}] Stream ended. Verifying complete size...")
            if total_size > 0 and downloaded >= total_size:
                part_path.rename(out_path)
                print(f"[{file_name}] Download COMPLETE! Saved to {out_path} ({out_path.stat().st_size / (1024**3):.2f} GB)")
                return out_path

        except Exception as e:
            print(f"\n[Attempt {attempt}] Connection error: {e}. Retrying in 3s...", flush=True)
            time.sleep(3)

    if part_path.exists() and not out_path.exists():
        part_path.rename(out_path)
    return out_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="painter-by-numbers")
    ap.add_argument("--file", required=True)
    ap.add_argument("--dest", default="/root/Workspace/data/wikiart")
    ap.add_argument("--user", default="xarcher")
    ap.add_argument("--key", default="555f4dbbcb1be1c3acf4e3809e42054f")
    args = ap.parse_args()

    download_file_resumable(args.comp, args.file, Path(args.dest), args.user, args.key)
