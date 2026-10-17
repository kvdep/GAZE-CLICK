import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from evaluate_video_benchmark import evaluate_video

video = "mvp/benchmarks/benchmark_test_20261014_115554.mp4"
csv_meta = "mvp/benchmarks/benchmark_test_20261014_115554.csv"

ckpt_010 = "mvp/experiments/yolo26n_gaze_net_yolo26n/run_010_20261015_020142/checkpoints/best_model.pt"
ckpt_011 = "mvp/experiments/yolo26n_gaze_net_yolo26n/run_011_20261015_110606/checkpoints/best_model.pt"

print("EVALUATING RUN_010 ON USER VIDEO:")
r010 = evaluate_video(video, csv_meta, checkpoint_path=ckpt_010)
print("Run_010 Mean Ang Err:", round(r010["angular_err_mean_deg"], 3), "deg")
print("Run_010 Median Ang Err:", round(r010["angular_err_median_deg"], 3), "deg")
print("Run_010 Corr X:", round(r010["corr_x"], 4))
print("Run_010 Corr Y:", round(r010["corr_y"], 4))
print("Run_010 Bias X:", round(r010["bias_x_px"], 1), "px")
print("Run_010 Bias Y:", round(r010["bias_y_px"], 1), "px")
print("Run_010 MAE X:", round(r010["mae_x_px"], 1), "px")
print("Run_010 MAE Y:", round(r010["mae_y_px"], 1), "px")

print("\nEVALUATING RUN_011 ON USER VIDEO:")
r011 = evaluate_video(video, csv_meta, checkpoint_path=ckpt_011)
print("Run_011 Mean Ang Err:", round(r011["angular_err_mean_deg"], 3), "deg")
print("Run_011 Median Ang Err:", round(r011["angular_err_median_deg"], 3), "deg")
print("Run_011 Corr X:", round(r011["corr_x"], 4))
print("Run_011 Corr Y:", round(r011["corr_y"], 4))
print("Run_011 Bias X:", round(r011["bias_x_px"], 1), "px")
print("Run_011 Bias Y:", round(r011["bias_y_px"], 1), "px")
print("Run_011 MAE X:", round(r011["mae_x_px"], 1), "px")
print("Run_011 MAE Y:", round(r011["mae_y_px"], 1), "px")
