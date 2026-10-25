import os, glob
import pandas as pd

oct17_files = sorted(glob.glob("mvp/benchmarks/benchmark_test_20261017_*.csv"))
print(f"Found {len(oct17_files)} Oct 17 sessions:")
for f in oct17_files:
    df = pd.read_csv(f)
    phases = df['phase'].value_counts()
    circle_n = phases.get('CircleTrajectory', 0)
    static_n = len(df) - circle_n
    det_rate = df['face_detected'].mean() * 100
    print(f"{os.path.basename(f)}: Total={len(df)}, Static={static_n}, Dynamic={circle_n}, FaceDet={det_rate:.1f}%")
