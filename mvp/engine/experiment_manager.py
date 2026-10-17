import os, json, glob, time, inspect, shutil, traceback, win32com.client, pandas as pd, numpy as np, torch
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from models.base import BaseGazeModel
from engine.inspector import ArchitectureInspector
from utils.logger import metric_log, attach_action_log_file
from utils.memory import MemoryManager

class ExperimentManager:
    def __init__(self, experiments_root: str = "mvp/experiments", timezone_offset_hours: int = 3) -> None:
        self.experiments_root = experiments_root
        self.timezone_info = timezone(timedelta(hours=timezone_offset_hours))
        os.makedirs(self.experiments_root, exist_ok=True)
        self.global_registry_path = os.path.join(self.experiments_root, "global_registry.json")

    def initialize_experiment(self, model: BaseGazeModel, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        arch_config = model.get_architecture_config()
        arch_type = arch_config.get("model_type", "gaze_model")
        backbone_tag = arch_config.get("backbone_name", arch_config.get("backbone", "standard"))
        architecture_name = f"{arch_type}_{backbone_tag}"
        architecture_dir = os.path.join(self.experiments_root, architecture_name)
        os.makedirs(architecture_dir, exist_ok=True)
        arch_report_path = os.path.join(architecture_dir, "architecture_report.md")
        arch_spec_path = os.path.join(architecture_dir, "architecture_spec.json")
        if not os.path.exists(arch_report_path) or not os.path.exists(arch_spec_path):
            inspection_results = ArchitectureInspector.inspect(model)
            ArchitectureInspector.generate_markdown_report(inspection_results, output_path=arch_report_path)
            ArchitectureInspector.save_specification(inspection_results, output_path=arch_spec_path)
        existing_runs = sorted(glob.glob(os.path.join(architecture_dir, "run_*")))
        next_run_index = len(existing_runs) + 1
        now_dt = datetime.now(self.timezone_info)
        timestamp_str = now_dt.strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{next_run_index:03d}_{timestamp_str}"
        run_dir = os.path.join(architecture_dir, run_name)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        run_action_log_path = os.path.join(run_dir, "action_log.txt")
        attach_action_log_file(run_action_log_path)
        try:
            original_model_file = inspect.getfile(model.__class__)
            copied_model_source_path = os.path.join(run_dir, "model_source.py")
            shutil.copy2(original_model_file, copied_model_source_path)
        except Exception:
            original_model_file = ""; copied_model_source_path = ""
        hyperparameter_diff: Dict[str, Dict[str, Any]] = {}
        if existing_runs:
            previous_run_dir = existing_runs[-1]
            previous_config_path = os.path.join(previous_run_dir, "config.json")
            if os.path.exists(previous_config_path):
                with open(previous_config_path, "r", encoding="utf-8") as f:
                    prev_config = json.load(f)
                prev_hyperparams = prev_config.get("hyperparameters", {})
                for k, v in hyperparameters.items():
                    if k in prev_hyperparams and prev_hyperparams[k] != v:
                        hyperparameter_diff[k] = {"previous": prev_hyperparams[k], "current": v}
                    elif k not in prev_hyperparams:
                        hyperparameter_diff[k] = {"previous": None, "current": v}
        config_payload = {
            "run_name": run_name,
            "architecture_name": architecture_name,
            "architecture_config": arch_config,
            "original_model_file": original_model_file,
            "copied_model_source_path": copied_model_source_path,
            "hyperparameters": hyperparameters,
            "hyperparameter_diff": hyperparameter_diff,
            "created_at": now_dt.isoformat()
        }
        config_file_path = os.path.join(run_dir, "config.json")
        with open(config_file_path, "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2, ensure_ascii=False)
        diff_file_path = os.path.join(run_dir, "hyperparameter_diff.json")
        with open(diff_file_path, "w", encoding="utf-8") as f:
            json.dump(hyperparameter_diff, f, indent=2, ensure_ascii=False)
        csv_log_path = os.path.join(run_dir, "training_log.csv")
        csv_headers = "epoch,train_loss,val_loss,val_angular_error_deg,val_pixel_mae,duration_seconds,learning_rate,ram_gb,vram_gb,timestamp_msk\n"
        with open(csv_log_path, "w", encoding="utf-8") as f:
            f.write(csv_headers)
        metric_log(f"Initialized experiment: {run_name} in [DIR:{architecture_dir}]")
        metric_log(f"Copied model source definition to: [ARTIFACT:{os.path.abspath(copied_model_source_path)}]")
        if hyperparameter_diff:
            diff_summary = ", ".join([f"{k}: {v['previous']} -> {v['current']}" for k, v in hyperparameter_diff.items()])
            metric_log(f"Hyperparameter modifications detected: {diff_summary}")
        return {
            "architecture_name": architecture_name,
            "architecture_dir": architecture_dir,
            "run_name": run_name,
            "run_dir": run_dir,
            "checkpoints_dir": checkpoints_dir,
            "config_path": config_file_path,
            "csv_log_path": csv_log_path,
            "model_source_file": copied_model_source_path,
            "original_model_file": original_model_file,
            "hyperparameter_diff": hyperparameter_diff
        }

    def resume_experiment(self, run_dir: str, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        run_dir = os.path.abspath(run_dir)
        architecture_dir = os.path.dirname(run_dir)
        architecture_name = os.path.basename(architecture_dir)
        run_name = os.path.basename(run_dir)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        run_action_log_path = os.path.join(run_dir, "action_log.txt")
        attach_action_log_file(run_action_log_path)
        config_file_path = os.path.join(run_dir, "config.json")
        copied_model_source_path = os.path.join(run_dir, "model_source.py")
        csv_log_path = os.path.join(run_dir, "training_log.csv")
        if not os.path.exists(csv_log_path):
            csv_headers = "epoch,train_loss,val_loss,val_angular_error_deg,val_pixel_mae,duration_seconds,learning_rate,ram_gb,vram_gb,timestamp_msk\n"
            with open(csv_log_path, "w", encoding="utf-8") as f:
                f.write(csv_headers)
        metric_log(f"Resumed experiment: {run_name} in [DIR:{architecture_dir}]")
        return {
            "architecture_name": architecture_name,
            "architecture_dir": architecture_dir,
            "run_name": run_name,
            "run_dir": run_dir,
            "checkpoints_dir": checkpoints_dir,
            "config_path": config_file_path,
            "csv_log_path": csv_log_path,
            "model_source_file": copied_model_source_path,
            "original_model_file": copied_model_source_path,
            "hyperparameter_diff": {}
        }

    def record_error(self, experiment_context: Dict[str, Any], exception: BaseException) -> str:
        run_dir = experiment_context.get("run_dir", "")
        run_name = experiment_context.get("run_name", "unknown_run")
        if not run_dir:
            run_dir = os.path.join(self.experiments_root, "failed_runs", run_name)
            os.makedirs(run_dir, exist_ok=True)
        error_log_path = os.path.join(run_dir, "error_traceback.log")
        tb_str = traceback.format_exc()
        now_dt = datetime.now(self.timezone_info).isoformat()
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"TIMESTAMP: {now_dt}\n")
            f.write(f"RUN_NAME: {run_name}\n")
            f.write(f"EXCEPTION_TYPE: {type(exception).__name__}\n")
            f.write(f"EXCEPTION_MESSAGE: {str(exception)}\n\n")
            f.write("TRACEBACK:\n")
            f.write(tb_str)
        error_shortcuts_dir = os.path.abspath("mvp/error_logs")
        os.makedirs(error_shortcuts_dir, exist_ok=True)
        shortcut_lnk_path = os.path.join(error_shortcuts_dir, f"{run_name}.lnk")
        try:
            ws = win32com.client.Dispatch("WScript.Shell")
            sc = ws.CreateShortcut(shortcut_lnk_path)
            sc.TargetPath = os.path.abspath(error_log_path)
            sc.Save()
        except Exception:
            pass
        shortcut_url_path = os.path.join(error_shortcuts_dir, f"{run_name}.url")
        try:
            with open(shortcut_url_path, "w", encoding="utf-8") as f_url:
                f_url.write(f"[InternetShortcut]\nURL=file:///{os.path.abspath(error_log_path).replace(os.sep, '/')}\n")
        except Exception:
            pass
        metric_log(f"Error trace written to: [ARTIFACT:{os.path.abspath(error_log_path)}]")
        metric_log(f"Error shortcut generated: [SHORTCUT:{shortcut_lnk_path}]")
        return error_log_path

    def record_epoch_log(self, csv_log_path: str, epoch_record: Dict[str, Any]) -> None:
        now_msk = datetime.now(self.timezone_info).strftime("%H:%M:%S")
        ram_gb = MemoryManager.get_free_ram_gb()
        vram_gb = MemoryManager.get_free_vram_gb()
        log_line = (
            f"{epoch_record['epoch']},"
            f"{epoch_record['train_loss']:.6f},"
            f"{epoch_record['val_loss']:.6f},"
            f"{epoch_record['val_angular_error_deg']:.4f},"
            f"{epoch_record['val_pixel_mae']:.4f},"
            f"{epoch_record['duration_seconds']:.2f},"
            f"{epoch_record['learning_rate']:.8f},"
            f"{ram_gb:.2f},"
            f"{vram_gb:.2f},"
            f"{now_msk}\n"
        )
        with open(csv_log_path, "a", encoding="utf-8") as f:
            f.write(log_line)

    def finalize_experiment(self, experiment_context: Dict[str, Any], training_results: Dict[str, Any], start_timestamp_mono: float) -> None:
        duration_total = time.monotonic() - start_timestamp_mono
        best_pt_path = os.path.abspath(training_results["best_checkpoint_path"])
        model_src_path = os.path.abspath(experiment_context["model_source_file"])
        is_interrupted = training_results.get("interrupted", False)
        summary_payload = {
            "run_name": experiment_context["run_name"],
            "architecture_name": experiment_context["architecture_name"],
            "status": "interrupted" if is_interrupted else "completed",
            "total_duration_seconds": round(duration_total, 2),
            "best_epoch": training_results["best_epoch"],
            "best_val_loss": round(training_results["best_val_loss"], 6) if not np.isnan(training_results["best_val_loss"]) else None,
            "test_loss": round(training_results["test_loss"], 6) if not np.isnan(training_results["test_loss"]) else None,
            "test_angular_error_deg": round(training_results["test_angular_error_deg"], 4) if not np.isnan(training_results["test_angular_error_deg"]) else None,
            "test_head_pitch_mae_deg": round(training_results["test_head_pitch_mae_deg"], 4) if "test_head_pitch_mae_deg" in training_results and not np.isnan(training_results["test_head_pitch_mae_deg"]) else None,
            "test_head_yaw_mae_deg": round(training_results["test_head_yaw_mae_deg"], 4) if "test_head_yaw_mae_deg" in training_results and not np.isnan(training_results["test_head_yaw_mae_deg"]) else None,
            "test_pixel_mae": round(training_results["test_pixel_mae"], 4) if not np.isnan(training_results["test_pixel_mae"]) else None,
            "total_epochs_executed": training_results["total_epochs_executed"],
            "weights_file_path": best_pt_path,
            "model_source_file": model_src_path,
            "original_model_file": os.path.abspath(experiment_context["original_model_file"]),
            "hyperparameter_diff": experiment_context["hyperparameter_diff"],
            "completed_at": datetime.now(self.timezone_info).isoformat()
        }
        summary_path = os.path.join(experiment_context["run_dir"], "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)
        arch_csv_path = os.path.join(experiment_context["architecture_dir"], "architecture_experiments.csv")
        flat_record = {
            "run_name": experiment_context["run_name"],
            "architecture": experiment_context["architecture_name"],
            "status": summary_payload["status"],
            "duration_s": summary_payload["total_duration_seconds"],
            "best_epoch": summary_payload["best_epoch"],
            "best_val_loss": summary_payload["best_val_loss"],
            "test_loss": summary_payload["test_loss"],
            "test_ang_deg": summary_payload["test_angular_error_deg"],
            "head_pitch_mae": summary_payload["test_head_pitch_mae_deg"],
            "head_yaw_mae": summary_payload["test_head_yaw_mae_deg"],
            "test_pixel_mae": summary_payload["test_pixel_mae"],
            "weights_path": best_pt_path,
            "model_source": model_src_path
        }
        arch_df = pd.DataFrame([flat_record])
        if os.path.exists(arch_csv_path):
            existing_arch_df = pd.read_csv(arch_csv_path)
            existing_arch_df = existing_arch_df[existing_arch_df["run_name"] != experiment_context["run_name"]]
            updated_arch_df = pd.concat([existing_arch_df, arch_df], ignore_index=True)
            updated_arch_df.to_csv(arch_csv_path, index=False)
        else:
            arch_df.to_csv(arch_csv_path, index=False)
        registry_data: List[Dict[str, Any]] = []
        if os.path.exists(self.global_registry_path):
            with open(self.global_registry_path, "r", encoding="utf-8") as f:
                try:
                    registry_data = json.load(f)
                except Exception:
                    registry_data = []
        registry_data = [item for item in registry_data if item.get("run_name") != experiment_context["run_name"]]
        registry_data.append(summary_payload)
        with open(self.global_registry_path, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2, ensure_ascii=False)
        metric_log(f"Finalized experiment: {experiment_context['run_name']}. Status: {summary_payload['status'].upper()}, Total duration: {duration_total:.2f}s, Best Val Loss: {summary_payload['best_val_loss']}")
        metric_log(f"Saved weights file (.pt): [WEIGHTS:{best_pt_path}]")
        metric_log(f"Saved model source definition (.py): [ARTIFACT:{model_src_path}]")

    def compare_experiments(self, architecture_name: Optional[str] = None) -> Tuple[pd.DataFrame, Any]:
        records: List[Dict[str, Any]] = []
        if os.path.exists(self.global_registry_path):
            with open(self.global_registry_path, "r", encoding="utf-8") as f:
                try:
                    raw_registry = json.load(f)
                    for item in raw_registry:
                        if architecture_name is None or item.get("architecture_name") == architecture_name:
                            records.append({
                                "run_name": item["run_name"],
                                "architecture": item["architecture_name"],
                                "val_loss": item["best_val_loss"],
                                "test_loss": item["test_loss"],
                                "angular_deg": item["test_angular_error_deg"],
                                "head_pitch_mae": item.get("test_head_pitch_mae_deg"),
                                "head_yaw_mae": item.get("test_head_yaw_mae_deg"),
                                "pixel_mae": item["test_pixel_mae"],
                                "best_epoch": item["best_epoch"],
                                "duration_s": item["total_duration_seconds"],
                                "weights_file": item.get("weights_file_path", ""),
                                "model_file": item.get("model_source_file", "")
                            })
                except Exception:
                    pass
        if not records:
            empty_df = pd.DataFrame(columns=["run_name", "architecture", "val_loss", "test_loss", "angular_deg", "head_pitch_mae", "head_yaw_mae", "pixel_mae", "best_epoch", "duration_s", "weights_file", "model_file"])
            return empty_df, empty_df.style
        dataframe = pd.DataFrame(records)
        format_dict = {
            "val_loss": "{:.4f}",
            "test_loss": "{:.4f}",
            "angular_deg": "{:.2f}°",
            "head_pitch_mae": "{:.2f}°",
            "head_yaw_mae": "{:.2f}°",
            "pixel_mae": "{:.1f} px",
            "duration_s": "{:.1f} s"
        }
        styler = dataframe.style.format(format_dict, na_rep="-")
        minimize_columns = ["val_loss", "test_loss", "angular_deg", "head_pitch_mae", "head_yaw_mae", "pixel_mae", "duration_s"]
        for col in minimize_columns:
            if col in dataframe.columns and dataframe[col].dropna().nunique() > 1:
                styler = styler.background_gradient(subset=[col], cmap="RdYlGn_r")
        return dataframe, styler
