import os, glob, time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def process_image(img_path: str) -> bool:
    try:
        norm_p = os.path.normpath(img_path)
        parts = norm_p.split(os.sep)
        subj = parts[-2]
        fname = parts[-1]
        out_dir = os.path.join("DATA", "columbia_gaze", "crops_224", subj)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            return True
        im = Image.open(img_path)
        crop_box = (1700, 900, 3500, 2700)
        im_cropped = im.crop(crop_box).resize((224, 224), Image.Resampling.BILINEAR)
        im_cropped.save(out_path, quality=95)
        return True
    except Exception as exc:
        print(f"Error processing {img_path}: {exc}")
        return False

def main() -> None:
    t0 = time.monotonic()
    all_images = glob.glob("DATA/columbia_gaze/Columbia Gaze Data Set/**/*.jpg", recursive=True)
    print(f"Starting pre-cropping of {len(all_images)} Columbia Gaze images...")
    workers = min(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(process_image, all_images))
    success_count = sum(1 for r in results if r)
    dt = time.monotonic() - t0
    print(f"Pre-cropping completed in {dt:.2f}s: {success_count}/{len(all_images)} images saved to DATA/columbia_gaze/crops_224/")

if __name__ == "__main__":
    main()
