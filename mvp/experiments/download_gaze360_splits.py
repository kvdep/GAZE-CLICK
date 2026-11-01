# -*- coding: utf-8 -*-
"""
Download Gaze360 split tar parts from HuggingFace mirror (immediately/Gaze360-split)
to E:\ВКР\Gaze360\splits with resume and retry support.
Total size: 30,548,756,480 bytes (28.45 GB).
"""
import os
import sys
import time
import requests

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

DEST_DIR = r"E:\ВКР\Gaze360\splits"
os.makedirs(DEST_DIR, exist_ok=True)

BASE_URL = "https://huggingface.co/datasets/immediately/Gaze360-split/resolve/main/"

PARTS = [
    ("Gaze360.tar.partaa", 5368709120),
    ("Gaze360.tar.partab", 5368709120),
    ("Gaze360.tar.partac", 5368709120),
    ("Gaze360.tar.partad", 5368709120),
    ("Gaze360.tar.partae", 5368709120),
    ("Gaze360.tar.partaf", 3705210880),
]

CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB chunks for fast I/O
MAX_RETRIES = 100

for filename, expected_size in PARTS:
    dest_path = os.path.join(DEST_DIR, filename)
    file_url = BASE_URL + filename
    
    retries = 0
    while True:
        current_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        if current_size >= expected_size:
            print(f"[SKIP] {filename} already complete ({current_size / (1024**3):.2f} GB)")
            break
        
        headers = {"User-Agent": "Mozilla/5.0"}
        mode = "ab" if current_size > 0 else "wb"
        if current_size > 0:
            headers["Range"] = f"bytes={current_size}-"
            print(f"[RESUME] {filename} from byte {current_size} ({current_size / (1024**2):.1f} / {expected_size / (1024**2):.1f} MB)...")
        else:
            print(f"[START] {filename} ({expected_size / (1024**3):.2f} GB)...")
            
        t0 = time.time()
        last_print = t0
        bytes_downloaded = current_size
        
        try:
            with requests.get(file_url, headers=headers, stream=True, timeout=30) as r:
                if r.status_code not in (200, 206):
                    print(f"[ERROR] HTTP {r.status_code} for {filename}, retrying in 5s...")
                    time.sleep(5)
                    retries += 1
                    if retries > MAX_RETRIES:
                        raise RuntimeError(f"Max retries exceeded for {filename}")
                    continue
                    
                with open(dest_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            bytes_downloaded += len(chunk)
                            now = time.time()
                            if now - last_print >= 5.0:
                                elapsed = now - t0
                                speed = (bytes_downloaded - current_size) / elapsed / (1024 * 1024) if elapsed > 0 else 0
                                pct = bytes_downloaded / expected_size * 100
                                print(f"[{filename} {pct:.1f}%] {bytes_downloaded / (1024**2):.1f}/{expected_size / (1024**2):.1f} MB ({speed:.2f} MB/s)")
                                last_print = now
                                
            final_size = os.path.getsize(dest_path)
            if final_size >= expected_size:
                print(f"[DONE] {filename} finished: {final_size} bytes (expected {expected_size})")
                break
        except (requests.exceptions.RequestException, Exception) as exc:
            current_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
            print(f"[WARN] Connection interrupted ({exc}). Current size: {current_size} bytes. Resuming in 3s...")
            time.sleep(3)
            retries += 1
            if retries > MAX_RETRIES:
                raise RuntimeError(f"Max retries exceeded for {filename}")

print("All Gaze360 split parts (partaa..partaf) downloaded and verified successfully!")
