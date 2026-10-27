# -*- coding: utf-8 -*-
import os, sys, time
import requests

def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    # Check if partially downloaded
    downloaded_bytes = 0
    if os.path.exists(dest_path):
        downloaded_bytes = os.path.getsize(dest_path)
        headers["Range"] = f"bytes={downloaded_bytes}-"
        print(f"Resuming download from byte {downloaded_bytes} ({downloaded_bytes/1024/1024:.2f} MB)...")
    else:
        print("Starting fresh download...")
        
    start_time = time.time()
    response = requests.get(url, headers=headers, stream=True, timeout=30)
    
    if response.status_code not in (200, 206):
        raise RuntimeError(f"Unexpected status code {response.status_code}")
        
    mode = "ab" if response.status_code == 206 else "wb"
    total_size = int(response.headers.get("content-length", 0)) + (downloaded_bytes if response.status_code == 206 else 0)
    
    curr = downloaded_bytes if response.status_code == 206 else 0
    last_print = time.time()
    
    with open(dest_path, mode) as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                curr += len(chunk)
                if time.time() - last_print > 5.0:
                    speed = (curr - downloaded_bytes) / (time.time() - start_time) / 1024 / 1024
                    pct = (curr / total_size * 100) if total_size > 0 else 0
                    print(f"[{pct:.1f}%] {curr/1024/1024:.1f}/{total_size/1024/1024:.1f} MB ({speed:.2f} MB/s)")
                    last_print = time.time()

    elapsed = time.time() - start_time
    print(f"Download completed successfully in {elapsed:.1f}s: {dest_path} ({os.path.getsize(dest_path)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    url = "https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/columbia_gaze_data_set.zip"
    dest = r"E:\ВКР\ColumbiaGaze\columbia_gaze_data_set.zip"
    download_file(url, dest)
