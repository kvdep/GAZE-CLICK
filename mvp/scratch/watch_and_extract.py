import os
import tarfile
import time

raw_dir = "DATA/rtgene_raw"
dest_dir = "DATA/rtgene/rtgene_dataset/extracted_images"
os.makedirs(dest_dir, exist_ok=True)

all_subjects = [f"s{i:03d}" for i in range(17)]
print(f"Watchdog started for RT-GENE subjects: {all_subjects}", flush=True)

while True:
    try:
        tars = [f for f in os.listdir(raw_dir) if f.endswith("_glasses.tar") and not f.endswith(".part")]
        for t in tars:
            tar_path = os.path.join(raw_dir, t)
            prefix = t.replace(".tar", "")
            target_subdir = os.path.join(dest_dir, prefix)
            
            if os.path.exists(target_subdir) and len(os.listdir(target_subdir)) > 0:
                try:
                    os.remove(tar_path)
                    print(f"Deleted already extracted tar: {tar_path}", flush=True)
                except Exception as e:
                    print(f"Error removing {tar_path}: {e}", flush=True)
                continue
                
            print(f"New tar detected: {t}. Starting extraction...", flush=True)
            t0 = time.monotonic()
            with tarfile.open(tar_path, "r") as archive:
                members = [
                    m for m in archive.getmembers()
                    if ("inpainted" in m.name and "face" in m.name and m.name.endswith(".png") and "face_after" not in m.name)
                ]
                print(f"Extracting {len(members)} face images from {t}...", flush=True)
                archive.extractall(path=dest_dir, members=members)
            dt = time.monotonic() - t0
            print(f"Extraction complete for {t}: {len(members)} files in {dt:.1f}s", flush=True)
            try:
                os.remove(tar_path)
                print(f"Deleted archive: {tar_path}", flush=True)
            except Exception as e:
                print(f"Error deleting {tar_path}: {e}", flush=True)
                
        existing_glasses_dirs = [d for d in os.listdir(dest_dir) if d.endswith("_glasses")]
        if len(existing_glasses_dirs) >= 17:
            print(f"All 17 glasses sessions extracted: {existing_glasses_dirs}", flush=True)
            break
            
    except Exception as exc:
        print(f"Watchdog iteration error: {exc}", flush=True)
        
    time.sleep(10)
