import os, sys, tarfile, time

raw_dir = "DATA/rtgene_raw"
dest_dir = "DATA/rtgene/rtgene_dataset/extracted_images"
os.makedirs(dest_dir, exist_ok=True)

targets = ["s004_glasses.tar", "s007_glasses.tar", "s008_glasses.tar"]
for t in targets:
    tar_path = os.path.join(raw_dir, t)
    if not os.path.exists(tar_path):
        print(f"File not found: {tar_path}")
        continue
    t_start = time.monotonic()
    print(f"Extracting inpainted images from {t}...")
    with tarfile.open(tar_path, "r") as archive:
        members = [
            m for m in archive.getmembers()
            if ("inpainted" in m.name and "face" in m.name and m.name.endswith(".png") and "face_after" not in m.name)
        ]
        print(f"Found {len(members)} matching inpainted face images. Extracting to {dest_dir}...")
        archive.extractall(path=dest_dir, members=members)
    dur = time.monotonic() - t_start
    print(f"Extracted {len(members)} files from {t} in {dur:.2f}s")
