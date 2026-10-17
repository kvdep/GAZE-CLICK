import gc, sys, os, psutil, torch
from contextlib import contextmanager
from typing import Generator, Tuple, Optional
from utils.logger import metric_log

class MemoryManager:
    @staticmethod
    def aggressive_cleanup() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect(); torch.cuda.synchronize()

    @staticmethod
    def get_free_ram_gb() -> float:
        return psutil.virtual_memory().available / (1024.0 ** 3)

    @staticmethod
    def get_free_vram_gb(device_index: int = 0) -> float:
        if not torch.cuda.is_available(): return 0.0
        return torch.cuda.mem_get_info(device_index)[0] / (1024.0 ** 3)

    @staticmethod
    def is_colab_environment() -> bool:
        return sys.platform.startswith("linux") and ("google.colab" in sys.modules or os.path.exists("/content"))

    @staticmethod
    def verify_runtime_environment(target_device: torch.device, min_required_vram_gb: float = 14.5) -> None:
        MemoryManager.aggressive_cleanup()
        if target_device.type == "cuda":
            current_free_vram = MemoryManager.get_free_vram_gb(target_device.index or 0)
            if MemoryManager.is_colab_environment() and current_free_vram < min_required_vram_gb:
                raise RuntimeError(f"Colab T4 invariant violated: Free VRAM is {current_free_vram:.2f} GB < Required {min_required_vram_gb:.2f} GB")
            metric_log(f"Runtime verified: Free VRAM = {current_free_vram:.2f} GB, Free RAM = {MemoryManager.get_free_ram_gb():.2f} GB", device_context="GPU")
        else:
            metric_log(f"Runtime verified: Free RAM = {MemoryManager.get_free_ram_gb():.2f} GB", device_context="CPU")

    @staticmethod
    def compute_dynamic_batch_size(sample_dimensions: Tuple[int, ...], parameter_count: int, target_device: torch.device, maximum_batch_size: int = 128, minimum_batch_size: int = 2) -> int:
        MemoryManager.aggressive_cleanup()
        if target_device.type == "cuda":
            current_free_bytes = torch.cuda.mem_get_info(target_device.index or 0)[0]
        else:
            current_free_bytes = psutil.virtual_memory().available
        reserved_safety_bytes = max(1024 ** 3, int(current_free_bytes * 0.25))
        usable_memory_bytes = max(0, current_free_bytes - reserved_safety_bytes)
        bytes_per_input_element = 4
        single_sample_elements = 1
        for dimension in sample_dimensions:
            single_sample_elements *= dimension
        single_sample_bytes = single_sample_elements * bytes_per_input_element
        forward_backward_overhead_factor = 28
        estimated_memory_per_batch_item = single_sample_bytes * forward_backward_overhead_factor
        model_and_optimizer_bytes = parameter_count * 16
        available_for_batch_allocation = max(0, usable_memory_bytes - model_and_optimizer_bytes)
        calculated_batch_capacity = int(available_for_batch_allocation // max(1, estimated_memory_per_batch_item))
        selected_batch_size = max(minimum_batch_size, min(maximum_batch_size, calculated_batch_capacity))
        return selected_batch_size

@contextmanager
def scoped_model_device(model_instance: torch.nn.Module, target_device: torch.device) -> Generator[torch.nn.Module, None, None]:
    try:
        model_instance.to(target_device)
        yield model_instance
    finally:
        model_instance.to("cpu")
        MemoryManager.aggressive_cleanup()
