import os, time, copy, torch, torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import DataLoader
from models.base import BaseGazeModel
from models.registry import ModelRegistry
from utils.logger import metric_log
from utils.memory import MemoryManager, scoped_model_device

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, restore_best_weights: bool = True, min_epochs: int = 0) -> None:
        self.patience = patience; self.min_delta = min_delta; self.restore_best_weights = restore_best_weights; self.min_epochs = min_epochs
        self.counter: int = 0; self.best_score: float = float("inf"); self.best_loss: float = float("inf"); self.early_stop_triggered: bool = False
        self.best_model_weights: Optional[Dict[str, Any]] = None; self.best_epoch: int = 0

    def step(self, current_metric: float, model: BaseGazeModel, current_epoch: int) -> bool:
        if current_metric < self.best_score - self.min_delta:
            self.best_score = current_metric; self.best_loss = current_metric; self.counter = 0; self.best_epoch = current_epoch
            if self.restore_best_weights:
                self.best_model_weights = copy.deepcopy(model.state_dict())
            return False
        if current_epoch >= self.min_epochs:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop_triggered = True
        return self.early_stop_triggered

    def restore(self, model: BaseGazeModel) -> None:
        if self.restore_best_weights and self.best_model_weights is not None:
            model.load_state_dict(self.best_model_weights)
            metric_log(f"Restored best model weights from epoch {self.best_epoch} with val_angular_error={self.best_score:.4f}°")

    def configure_state(self, best_score: float, best_loss: float, best_epoch: int, best_weights: Optional[Dict[str, Any]] = None, counter: int = 0) -> None:
        self.best_score = best_score
        self.best_loss = best_loss
        self.best_epoch = best_epoch
        if best_weights is not None:
            self.best_model_weights = copy.deepcopy(best_weights)
        self.counter = counter

class GazeTrainer:
    def __init__(
        self,
        model: BaseGazeModel,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        target_device: torch.device = torch.device("cpu"),
        minimum_epochs: int = 150,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_dir: Optional[str] = None,
        save_interval_minutes: float = 20.0,
        save_every_epoch: bool = False
    ) -> None:
        self.model = model; self.criterion = criterion; self.optimizer = optimizer; self.scheduler = scheduler
        self.target_device = target_device
        self.total_epochs = max(150, minimum_epochs)
        self.early_stopping = early_stopping or EarlyStopping(patience=15, min_delta=1e-4)
        self.checkpoint_dir = checkpoint_dir or "mvp/checkpoints"
        self.save_interval_minutes = save_interval_minutes
        self.save_every_epoch = save_every_epoch
        self.last_periodic_save_mono = time.monotonic()
        self.model.to("cpu")
        MemoryManager.aggressive_cleanup()

    def train_epoch(self, train_loader: DataLoader) -> float:
        total_epoch_loss = 0.0; batch_count = 0
        with scoped_model_device(self.model, self.target_device) as active_model:
            active_model.train()
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.target_device)
            for batch_data in train_loader:
                self.optimizer.zero_grad()
                batch_on_device = {
                    key: value.to(self.target_device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch_data.items()
                }
                predictions = active_model(batch_on_device["image"])
                loss_breakdown = self.criterion(predictions, batch_on_device)
                total_loss = loss_breakdown["total_loss"]
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(active_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_epoch_loss += total_loss.item()
                batch_count += 1
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to("cpu")
        return total_epoch_loss / max(1, batch_count)

    def evaluate(self, eval_loader: DataLoader) -> Tuple[float, float, float, float, float]:
        total_eval_loss = 0.0; total_angular_error = 0.0; total_hp_mae = 0.0; total_hy_mae = 0.0; total_pixel_error = 0.0
        sample_count = 0; screen_sample_count = 0; batch_count = 0
        with scoped_model_device(self.model, self.target_device) as active_model:
            active_model.eval()
            with torch.no_grad():
                for batch_data in eval_loader:
                    batch_on_device = {
                        key: value.to(self.target_device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch_data.items()
                    }
                    predictions = active_model(batch_on_device["image"])
                    loss_breakdown = self.criterion(predictions, batch_on_device)
                    total_eval_loss += loss_breakdown["total_loss"].item()
                    batch_size = len(batch_data["gaze_angles"])
                    pred_vector = predictions["gaze_vector"]
                    target_vector = batch_on_device["gaze_vector"]
                    batch_angular_errors = BaseGazeModel.compute_angular_error_degrees(pred_vector, target_vector)
                    total_angular_error += torch.sum(batch_angular_errors).item()
                    if "head_pitch_rad" in predictions and "head_angles" in batch_on_device:
                        pred_hp = predictions["head_pitch_rad"] * (180.0 / 3.141592653589793)
                        tgt_hp = batch_on_device["head_angles"][:, 0] * (180.0 / 3.141592653589793)
                        pred_hy = predictions["head_yaw_rad"] * (180.0 / 3.141592653589793)
                        tgt_hy = batch_on_device["head_angles"][:, 1] * (180.0 / 3.141592653589793)
                        total_hp_mae += torch.sum(torch.abs(pred_hp - tgt_hp)).item()
                        total_hy_mae += torch.sum(torch.abs(pred_hy - tgt_hy)).item()
                    if "screen_coordinates" in predictions and "screen_coords" in batch_on_device:
                        valid_mask = batch_on_device.get("has_screen_coords", torch.zeros(batch_size, dtype=torch.bool, device=self.target_device))
                        if valid_mask.any():
                            pixel_mae = BaseGazeModel.compute_pixel_mae(predictions["screen_coordinates"], batch_on_device["screen_coords"], valid_mask)
                            mask_sum = torch.sum(valid_mask).item()
                            total_pixel_error += pixel_mae.item() * mask_sum
                            screen_sample_count += mask_sum
                    sample_count += batch_size
                    batch_count += 1
        mean_eval_loss = total_eval_loss / max(1, batch_count)
        mean_angular_deg = total_angular_error / max(1, sample_count)
        mean_hp_mae = total_hp_mae / max(1, sample_count)
        mean_hy_mae = total_hy_mae / max(1, sample_count)
        mean_pixel_mae = total_pixel_error / max(1, screen_sample_count) if screen_sample_count > 0 else 0.0
        return mean_eval_loss, mean_angular_deg, mean_hp_mae, mean_hy_mae, mean_pixel_mae

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, log_callback: Optional[Any] = None, start_epoch: int = 1) -> Dict[str, Any]:
        MemoryManager.verify_runtime_environment(self.target_device)
        dev_tag = "GPU" if self.target_device.type == "cuda" else "CPU"
        metric_log(f"Starting training run: total_epochs={self.total_epochs}, patience={self.early_stopping.patience}, save_interval_min={self.save_interval_minutes}", device_context=dev_tag)
        training_history: List[Dict[str, Any]] = []
        best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        latest_checkpoint_path = os.path.join(self.checkpoint_dir, "latest_model.pt")
        interrupted_checkpoint_path = os.path.join(self.checkpoint_dir, "interrupted_model.pt")
        last_val_loss = float("inf"); last_val_ang = float("nan"); last_val_hp = float("nan"); last_val_hy = float("nan"); last_val_pix = float("nan")
        interrupted = False
        try:
            for epoch in range(start_epoch, self.total_epochs + 1):
                epoch_start_time = time.monotonic()
                train_loss = self.train_epoch(train_loader)
                val_loss, val_ang_deg, val_hp_mae, val_hy_mae, val_pix_mae = self.evaluate(val_loader)
                last_val_loss = val_loss; last_val_ang = val_ang_deg; last_val_hp = val_hp_mae; last_val_hy = val_hy_mae; last_val_pix = val_pix_mae
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_ang_deg)
                    else:
                        self.scheduler.step()
                epoch_duration = time.monotonic() - epoch_start_time
                lr_hd = self.optimizer.param_groups[-1]["lr"]
                lr_bb = self.optimizer.param_groups[0]["lr"]
                lr_str = f"LR bb={lr_bb:.6f}, hd={lr_hd:.6f}" if len(self.optimizer.param_groups) > 1 else f"LR: {lr_hd:.6f}"
                metric_log(f"Epoch {epoch}/{self.total_epochs} | L Train: {train_loss:.4f} | L Val: {val_loss:.4f} | Gaze Val: {val_ang_deg:.2f}° | Head P/Y: {val_hp_mae:.2f}°/{val_hy_mae:.2f}° | {lr_str}", device_context=dev_tag)
                epoch_record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_angular_error_deg": val_ang_deg,
                    "val_head_pitch_mae_deg": val_hp_mae,
                    "val_head_yaw_mae_deg": val_hy_mae,
                    "val_pixel_mae": val_pix_mae,
                    "duration_seconds": epoch_duration,
                    "learning_rate": lr_hd
                }
                training_history.append(epoch_record)
                if log_callback is not None:
                    log_callback(epoch_record)
                if val_ang_deg <= self.early_stopping.best_score:
                    ModelRegistry.save_checkpoint(
                        best_checkpoint_path,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch=epoch,
                        metrics={"val_loss": val_loss, "val_angular_error_deg": val_ang_deg, "val_head_pitch_mae_deg": val_hp_mae, "val_head_yaw_mae_deg": val_hy_mae}
                    )
                    metric_log(f"Saved new best model checkpoint (Gaze Error: {val_ang_deg:.2f}°): [WEIGHTS:{os.path.abspath(best_checkpoint_path)}]", device_context=dev_tag)
                elapsed_save_min = (time.monotonic() - self.last_periodic_save_mono) / 60.0
                if self.save_every_epoch or elapsed_save_min >= self.save_interval_minutes:
                    ModelRegistry.save_checkpoint(
                        latest_checkpoint_path,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch=epoch,
                        metrics={"val_loss": val_loss, "val_angular_error_deg": val_ang_deg}
                    )
                    self.last_periodic_save_mono = time.monotonic()
                    metric_log(f"Periodic checkpoint synchronized: [WEIGHTS:{os.path.abspath(latest_checkpoint_path)}]", device_context=dev_tag)
                stop_now = self.early_stopping.step(val_ang_deg, self.model, epoch)
                if stop_now:
                    metric_log(f"Early stopping triggered at epoch {epoch}. Restoring best weights.", device_context=dev_tag)
                    break
        except KeyboardInterrupt:
            interrupted = True
            current_epoch_idx = len(training_history) + start_epoch - 1
            metric_log(f"Graceful shutdown initiated by user interrupt at epoch {current_epoch_idx}. Preserving weights and architecture.", device_context=dev_tag)
            ModelRegistry.save_checkpoint(
                interrupted_checkpoint_path,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch=current_epoch_idx,
                metrics={"val_loss": last_val_loss, "val_angular_error_deg": last_val_ang, "interrupted": True}
            )
            metric_log(f"Emergency interrupted checkpoint serialized: [WEIGHTS:{os.path.abspath(interrupted_checkpoint_path)}]", device_context=dev_tag)
        except Exception as exc:
            current_epoch_idx = len(training_history) + start_epoch - 1
            ModelRegistry.save_checkpoint(
                interrupted_checkpoint_path,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch=current_epoch_idx,
                metrics={"val_loss": last_val_loss, "interrupted": True, "error": str(exc)}
            )
            metric_log(f"Exception encountered during training. Weights salvaged: [WEIGHTS:{os.path.abspath(interrupted_checkpoint_path)}]", device_context=dev_tag)
            raise exc
        if not interrupted:
            self.early_stopping.restore(self.model)
            test_loss, test_ang_deg, test_hp_mae, test_hy_mae, test_pix_mae = self.evaluate(test_loader)
            metric_log(f"L Test: {test_loss:.4f} | Gaze Error: {test_ang_deg:.2f}° | Head Pitch MAE: {test_hp_mae:.2f}° | Head Yaw MAE: {test_hy_mae:.2f}° | Screen MAE: {test_pix_mae:.1f} px", device_context=dev_tag)
            effective_checkpoint = best_checkpoint_path if os.path.exists(best_checkpoint_path) else latest_checkpoint_path
        else:
            test_loss = float("nan"); test_ang_deg = float("nan"); test_hp_mae = float("nan"); test_hy_mae = float("nan"); test_pix_mae = float("nan")
            effective_checkpoint = interrupted_checkpoint_path
        return {
            "interrupted": interrupted,
            "best_epoch": self.early_stopping.best_epoch if not interrupted else len(training_history),
            "best_val_loss": self.early_stopping.best_loss if not interrupted else last_val_loss,
            "test_loss": test_loss,
            "test_angular_error_deg": test_ang_deg,
            "test_head_pitch_mae_deg": test_hp_mae,
            "test_head_yaw_mae_deg": test_hy_mae,
            "test_pixel_mae": test_pix_mae,
            "total_epochs_executed": len(training_history),
            "history": training_history,
            "best_checkpoint_path": effective_checkpoint,
            "interrupted_checkpoint_path": interrupted_checkpoint_path if interrupted else ""
        }
