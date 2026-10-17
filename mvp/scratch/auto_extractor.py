import os
import tarfile
import time

raw_dir = "DATA/rtgene_raw"
dest_dir = "DATA/rtgene/rtgene_dataset/extracted_images"
os.makedirs(dest_dir, exist_ok=True)

targets = [f for f in os.listdir(raw_dir) if f.endswith("_glasses.tar") and not f.endswith(".part")]
print(f"Candidate tar archives: {targets}")

for t in targets:
    tar_path = os.path.join(raw_dir, t)
    prefix = t.replace(".tar", "")
    target_subdir = os.path.join(dest_dir, prefix)
    
    if os.path.exists(target_subdir) and len(os.listdir(target_subdir)) > 0:
        print(f"Already extracted: {prefix} ({len(os.listdir(target_subdir))} items). Removing archive: {t}")
        try:
            os.remove(tar_path)
            print(f"Removed archive: {tar_path}")
        except Exception as e:
            print(f"Error removing {tar_path}: {e}")
        continue

    print(f"Extracting inpainted images from {t}...")
    t_start = time.monotonic()
    try:
        with tarfile.open(tar_path, "r") as archive:
            members = [
                m for m in archive.getmembers()
                if ("inpainted" in m.name and "face" in m.name and m.name.endswith(".png") and "face_after" not in m.name)
            ]
            print(f"Found {len(members)} matching inpainted face images. Extracting to {dest_dir}...")
            archive.extractall(path=dest_dir, members=members)
        dur = time.monotonic() - t_start
        print(f"Extracted {len(members)} files from {t} in {dur:.2f}s")
        os.remove(tar_path)
        print(f"Removed archive after extraction: {tar_path}")
    except Exception as e:
        print(f"Failed extraction for {t}: {e}")
