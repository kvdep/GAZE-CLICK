import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from test_calibrated_benchmark import run_calibrated_benchmark

v = "mvp/benchmarks/benchmark_test_20261014_115554.mp4"
m = "mvp/benchmarks/benchmark_test_20261014_115554.parquet"

checkpoints = [
    "mvp/experiments/yolo26n_gaze_net_yolo26n/run_010_20261015_020142/checkpoints/best_model.pt",
    "mvp/experiments/yolo26n_gaze_net_yolo26n/run_013_20261016_182944/checkpoints/latest_model.pt",
    "mvp/experiments/yolo26n_gaze_net_yolo26n/run_014_20261017_121921/checkpoints/best_model.pt",
    "mvp/experiments/yolo26n_gaze_net_yolo26n/run_014_20261017_121921/checkpoints/latest_model.pt"
]

for cp in checkpoints:
    if os.path.exists(cp):
        res = run_calibrated_benchmark(cp, v, m)
        run_name = cp.split("/")[-3]
        ckpt_name = cp.split("/")[-1]
        print(f"=== {run_name} {ckpt_name} ===")
        print(f"  Med Ang: {res['angular_err_median_deg']:.2f} deg, Mean Ang: {res['angular_err_mean_deg']:.2f} deg")
        print(f"  Med mm:  {res['screen_err_median_mm']:.1f} mm, Mean mm:  {res['screen_err_mean_mm']:.1f} mm")
        print(f"  MAE X:   {res['mae_x_px']:.1f} px, MAE Y:   {res['mae_y_px']:.1f} px")
        print(f"  Rx: {res['corr_x']:.3f}, Ry: {res['corr_y']:.3f}, R2: {res['calib_r2']:.3f}")
