#!/usr/bin/env python3
"""Download the WEAVE large-file archives (datasets + result images) and unpack them.

The archives live on Google Drive and are listed in scripts/data_links.json
(identical copies are mirrored at the URLs in README.md -> "Data & Results").

Examples
--------
    python scripts/download_data.py                  # download + unpack all
    python scripts/download_data.py --only data      # only data.tar
    python scripts/download_data.py --root .         # unpack under the repo root (default)
    python scripts/download_data.py --keep           # keep archives after unpack
    python scripts/download_data.py --verify         # only verify already-downloaded archives
    python scripts/download_data.py --check          # print what would be downloaded

Requirements: Python 3.8+ (standard library only) and, if available, the system
`tar` command. On Windows 10+ / macOS / most Linux distros tar is bundled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_links.json")

# Google Drive hosts large files behind a "virus scan" interstitial page.
_CONFIRM_RE = re.compile(r"confirm=([0-9A-Za-z_\-]+)")
_UUID_RE = re.compile(r"^[0-9A-Za-z_\-]{20,}$")


def load_manifest():
    with open(MANIFEST, "r", encoding="utf-8") as fh:
        return json.load(fh)


def format_size(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _open_gdrive_url(url, expected_size):
    """Open a Google Drive download URL, following the large-file confirm flow.

    Returns (response, filename_hint). The caller is responsible for closing.
    """
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    current = url
    for _ in range(8):
        req = urllib.request.Request(current, headers={"User-Agent": "Mozilla/5.0"})
        try:
            resp = opener.open(req, timeout=60)
        except urllib.error.HTTPError as err:
            if err.code in (429, 500, 502, 503, 504):
                time.sleep(3)
                continue
            raise
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return resp, None
        text = resp.read(1 << 20).decode("utf-8", errors="replace")
        resp.close()
        if "Too many users" in text:
            raise RuntimeError("Google Drive rate limit: too many users are viewing this file. Try again later.")
        # New-style flow: the interstitial embeds a <form> whose hidden inputs
        # carry id/export/confirm/uuid and whose action is drive.usercontent.google.com.
        m = re.search(r'action="([^"]+)"', text)
        if m and "drive.usercontent.google.com" in m.group(1):
            action = m.group(1).replace("&amp;", "&")
            inputs = re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', text)
            inputs += [(b, a) for a, b in re.findall(r'<input[^>]*value="([^"]*)"[^>]*name="([^"]+)"', text)]
            query = urllib.parse.urlencode(inputs)
            current = action + ("&" if "?" in action else "?") + query
            continue
        # Legacy flow: the interstitial carries a one-time confirm token.
        m = _CONFIRM_RE.search(text)
        if m and _UUID_RE.match(m.group(1)):
            current = "https://drive.google.com/uc?confirm={}&id={}".format(
                m.group(1), urllib.parse.parse_qs(urllib.parse.urlparse(current).query).get("id", [""])[0]
            )
            continue
        raise RuntimeError("Could not resolve Google Drive download URL")
    raise RuntimeError("Could not resolve Google Drive download URL")


def download_file(name, url, dest, expected_size):
    tmp = dest + ".part"
    if os.path.exists(dest) and expected_size and os.path.getsize(dest) == expected_size:
        print(f"[ok] {name} already complete")
        return
    print(f"[dl] {name} ({format_size(expected_size or 0)}) -> {dest}")
    resp, _ = _open_gdrive_url(url, expected_size)
    done = 0
    with open(tmp, "wb") as out:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if done % (64 << 20) < (1 << 20):
                print(f"    {format_size(done)}", flush=True)
    resp.close()
    if expected_size and done != expected_size:
        raise RuntimeError(f"{name}: size mismatch (expected {expected_size}, got {done})")
    os.replace(tmp, dest)
    print(f"[ok] {name} downloaded")


def extract_archive(path, root, top_dir):
    """Extract a .tar archive into root, restoring the top-level directory."""
    dest = os.path.join(root, top_dir)
    print(f"[unpack] {os.path.basename(path)} -> {root} (restores '{top_dir}/')")
    tar = shutil.which("tar")
    if tar:
        subprocess.check_call([tar, "-xf", path, "-C", root])
    else:
        with tarfile.open(path, "r") as tf:
            members = []
            prefix = top_dir + "/"
            for m in tf.getmembers():
                if not (m.name == top_dir or m.name.startswith(prefix)):
                    raise RuntimeError(f"archive contains unexpected path: {m.name}")
            tf.extractall(root, members=members)
    if not os.path.isdir(dest):
        raise RuntimeError(f"extraction did not produce {dest}")
    print(f"[ok] {top_dir}/ ready")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", default=None, choices=None,
                    help="only process the given archive names (e.g. data results)")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="directory to unpack into (default: repo root)")
    ap.add_argument("--dl-dir", default=None,
                    help="directory for the .tar archives (default: a 'downloads' dir in --root)")
    ap.add_argument("--keep", action="store_true", help="do not delete archives after unpacking")
    ap.add_argument("--verify", action="store_true", help="only verify sizes/hashes of existing archives")
    ap.add_argument("--check", action="store_true", help="print the manifest and exit")
    args = ap.parse_args()

    manifest = load_manifest()
    if args.check:
        for f in manifest["files"]:
            print(f'{f["name"]:20} {format_size(f["size"])}  ->  {f["unpack"]}/   url={f.get("url")}')
        return

    root = os.path.abspath(args.root)
    dl_dir = os.path.abspath(args.dl_dir) if args.dl_dir else os.path.join(root, "downloads")
    os.makedirs(dl_dir, exist_ok=True)

    wanted = set(args.only) if args.only else None
    missing_url = [f["name"] for f in manifest["files"] if not f.get("url")]
    if missing_url:
        print(f"error: manifest missing URLs for: {', '.join(missing_url)}", file=sys.stderr)
        sys.exit(1)

    for entry in manifest["files"]:
        if wanted and entry["name"] not in wanted and entry["unpack"] not in wanted:
            continue
        path = os.path.join(dl_dir, entry["name"])
        if args.verify:
            if not os.path.exists(path):
                print(f"[!!] {entry['name']}: missing")
                continue
            ok = True
            if entry.get("size") and os.path.getsize(path) != entry["size"]:
                print(f"[!!] {entry['name']}: size mismatch")
                ok = False
            if ok and entry.get("sha256"):
                print(f"[hash] {entry['name']} ...", end=" ", flush=True)
                actual = sha256_file(path)
                print("OK" if actual == entry["sha256"] else f"MISMATCH ({actual})")
                ok = ok and actual == entry["sha256"]
            if ok:
                print(f"[ok] {entry['name']}: verified")
            continue
        download_file(entry["name"], entry["url"], path, entry.get("size"))
        if entry.get("sha256"):
            print(f"[hash] {entry['name']} ...", end=" ", flush=True)
            actual = sha256_file(path)
            if actual != entry["sha256"]:
                print("MISMATCH, please re-download")
                sys.exit(1)
            print("OK")
        extract_archive(path, root, entry["unpack"])
        if not args.keep:
            os.remove(path)
            print(f"[cleanup] removed {entry['name']}")

    print("\nAll archives processed.")


if __name__ == "__main__":
    main()
