import os, glob
import pandas as pd
import cv2

bench_dir = "mvp/benchmarks"
csv_files = sorted(glob.glob(os.path.join(bench_dir, "*.csv")))

print(f"Total CSV files found: {len(csv_files)}")
pairs = []
for c in csv_files:
    base = c[:-4]
    m = base + ".mp4"
    if os.path.exists(m):
        try:
            df = pd.read_csv(c)
            cap = cv2.VideoCapture(m)
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            pairs.append((os.path.basename(base), len(df), fc, fps, w, h))
        except Exception as e:
            print(f"Error reading {base}: {e}")

print(f"{'Session Name':<45} | {'CSV Rows':<8} | {'Video Frames':<12} | {'FPS':<4} | {'Res'}")
print("-" * 85)
for name, rows, fc, fps, w, h in pairs:
    print(f"{name:<45} | {rows:<8} | {fc:<12} | {fps:<4.1f} | {w}x{h}")
