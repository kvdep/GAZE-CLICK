import os, sys, time, psutil, torch, inspect
from datetime import datetime, timezone, timedelta
from typing import Optional, List

class MetricLogger:
    def __init__(self, timezone_offset_hours: int = 3, global_log_directory: Optional[str] = None) -> None:
        self.timezone_info = timezone(timedelta(hours=timezone_offset_hours)); self.previous_timestamp: float = time.monotonic()
        self.log_file_paths: List[str] = []
        if global_log_directory is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            global_log_directory = os.path.join(base_dir, "logs")
        os.makedirs(global_log_directory, exist_ok=True)
        global_log_file = os.path.join(global_log_directory, "action_log.txt")
        self.add_file_handler(global_log_file)
        try:
            sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
        except Exception:
            pass

    def add_file_handler(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        if file_path not in self.log_file_paths:
            self.log_file_paths.append(file_path)

    @staticmethod
    def _extract_caller_info() -> str:
        frame = inspect.currentframe()
        while frame:
            fname = frame.f_code.co_filename
            base = os.path.basename(fname)
            if base != "logger.py":
                caller_name = "cli" if base == "<string>" else base
                return f"[{caller_name}:{frame.f_lineno}]"
            frame = frame.f_back
        return "[caller:unknown]"

    def log(self, message: str, device_context: Optional[str] = None) -> None:
        current_monotonic = time.monotonic()
        elapsed_seconds = int(current_monotonic - self.previous_timestamp); self.previous_timestamp = current_monotonic
        elapsed_hours = elapsed_seconds // 3600; elapsed_minutes = (elapsed_seconds % 3600) // 60; elapsed_secs = elapsed_seconds % 60
        elapsed_formatted = f"{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_secs:02d}"
        current_msk_time = datetime.now(self.timezone_info).strftime("%H:%M:%S")
        available_ram_gb = psutil.virtual_memory().available / (1024.0 ** 3)
        cuda_is_active = torch.cuda.is_available()
        resolved_device = "GPU" if (device_context == "GPU" or (device_context is None and cuda_is_active)) else "CPU"
        caller_tag = self._extract_caller_info()
        if resolved_device == "GPU" and cuda_is_active:
            free_vram_bytes, _ = torch.cuda.mem_get_info(0)
            available_vram_gb = free_vram_bytes / (1024.0 ** 3)
            log_line = f"[{current_msk_time}] [{elapsed_formatted}] [GPU] [{available_ram_gb:.1f}] [{available_vram_gb:.1f}] {caller_tag} {message}"
        else:
            log_line = f"[{current_msk_time}] [{elapsed_formatted}] [CPU] [{available_ram_gb:.1f}] [0.0] {caller_tag} {message}"
        print(log_line, flush=True)
        for log_path in self.log_file_paths:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(log_line + "\n")
            except Exception:
                pass

GLOBAL_LOGGER = MetricLogger()

def metric_log(message: str, device_context: Optional[str] = None) -> None:
    GLOBAL_LOGGER.log(message, device_context=device_context)

def attach_action_log_file(file_path: str) -> None:
    GLOBAL_LOGGER.add_file_handler(file_path)

