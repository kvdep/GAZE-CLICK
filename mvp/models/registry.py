import os, inspect, torch
from typing import Dict, Any, Type, Optional, Tuple
from models.base import BaseGazeModel
from models.l2cs_net import L2CSGazeNet
from utils.logger import metric_log

class ModelRegistry:
    _registry_table: Dict[str, Type[BaseGazeModel]] = {
        "l2cs_gaze_net": L2CSGazeNet
    }

    @classmethod
    def register_architecture(cls, architecture_identifier: str, model_class: Type[BaseGazeModel]) -> None:
        cls._registry_table[architecture_identifier] = model_class

    @classmethod
    def register(cls, architecture_identifier: str):
        def decorator(subclass: Type[BaseGazeModel]):
            cls._registry_table[architecture_identifier] = subclass
            return subclass
        return decorator

    @classmethod
    def create_model(cls, architecture_type: str = "l2cs_gaze_net", **model_arguments: Any) -> BaseGazeModel:
        if architecture_type not in cls._registry_table:
            raise KeyError(f"Architecture '{architecture_type}' is not registered. Available: {list(cls._registry_table.keys())}")
        model_constructor = cls._registry_table[architecture_type]
        instantiated_model = model_constructor(**model_arguments)
        return instantiated_model

    @classmethod
    def create_from_architecture_config(cls, architecture_config: Dict[str, Any]) -> BaseGazeModel:
        copied_config = dict(architecture_config)
        architecture_type = copied_config.pop("model_type")
        copied_config.pop("feature_dimension", None)
        return cls.create_model(architecture_type=architecture_type, **copied_config)

    @classmethod
    def save_checkpoint(cls, checkpoint_file_path: str, model: BaseGazeModel, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[Any] = None, epoch: int = 0, hyperparameters: Optional[Dict[str, Any]] = None, metrics: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_file_path)), exist_ok=True)
        try:
            source_file = inspect.getfile(model.__class__)
            source_code = inspect.getsource(model.__class__)
        except Exception:
            source_file = ""; source_code = ""
        checkpoint_payload = {
            "architecture_config": model.get_architecture_config(),
            "model_class_name": model.__class__.__name__,
            "model_source_file": source_file,
            "model_source_code": source_code,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "hyperparameters": hyperparameters or {},
            "metrics": metrics or {}
        }
        torch.save(checkpoint_payload, checkpoint_file_path)
        metric_log(f"Saved model checkpoint with architecture to: {os.path.abspath(checkpoint_file_path)}")

    @classmethod
    def load_checkpoint(cls, checkpoint_file_path: str, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[Any] = None, target_device: str = "cpu") -> Tuple[BaseGazeModel, Dict[str, Any]]:
        if not os.path.exists(checkpoint_file_path):
            raise FileNotFoundError(f"Checkpoint weights file not found: {checkpoint_file_path}")
        metric_log(f"Loading model weights from file: {os.path.abspath(checkpoint_file_path)}")
        loaded_payload = torch.load(checkpoint_file_path, map_location=target_device)
        if "architecture_config" not in loaded_payload:
            raise KeyError(f"Checkpoint file {checkpoint_file_path} does not contain architecture_config required for reconstruction")
        architecture_config = loaded_payload["architecture_config"]
        reconstructed_model = cls.create_from_architecture_config(architecture_config)
        reconstructed_model.load_state_dict(loaded_payload["model_state_dict"], strict=False)
        reconstructed_model.to(target_device)
        if optimizer is not None and loaded_payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(loaded_payload["optimizer_state_dict"])
        if scheduler is not None and loaded_payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(loaded_payload["scheduler_state_dict"])
        metric_log(f"Loaded model weights from file: {os.path.abspath(checkpoint_file_path)}")
        return reconstructed_model, loaded_payload
