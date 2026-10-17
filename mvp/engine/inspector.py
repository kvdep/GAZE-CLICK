import os, json, time, numpy as np, torch, torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from utils.logger import metric_log

class ArchitectureInspector:
    @staticmethod
    def inspect(model: nn.Module, input_tensor_shape: Tuple[int, ...] = (1, 3, 224, 224), target_device: torch.device = torch.device("cpu"), benchmark_runs: int = 20) -> Dict[str, Any]:
        model.eval(); model.to(target_device)
        total_parameters = sum(param.numel() for param in model.parameters())
        trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
        non_trainable_parameters = total_parameters - trainable_parameters
        buffer_elements = sum(buf.numel() for buf in model.buffers())
        model_memory_mb = float((total_parameters + buffer_elements) * 4) / (1024.0 ** 2)
        layer_records: List[Dict[str, Any]] = []
        hooks = []
        total_flops = 0

        def create_hook(module_name: str):
            def hook_fn(module: nn.Module, inputs: Tuple[Any, ...], output: Any):
                nonlocal total_flops
                in_shape = list(inputs[0].shape) if inputs and hasattr(inputs[0], "shape") else []
                out_shape = list(output.shape) if hasattr(output, "shape") else (list(output[list(output.keys())[0]].shape) if isinstance(output, dict) else [])
                module_params = sum(p.numel() for p in module.parameters(recurse=False))
                flops = 0
                if isinstance(module, nn.Conv2d):
                    batch, _, out_h, out_w = out_shape if len(out_shape) == 4 else (1, 1, 1, 1)
                    k_h, k_w = module.kernel_size; in_c = module.in_channels; out_c = module.out_channels
                    flops = int(2 * batch * in_c * out_c * k_h * k_w * out_h * out_w / module.groups)
                elif isinstance(module, nn.Linear):
                    batch = in_shape[0] if in_shape else 1
                    flops = int(2 * batch * module.in_features * module.out_features)
                total_flops += flops
                layer_records.append({
                    "name": module_name,
                    "type": module.__class__.__name__,
                    "input_shape": in_shape,
                    "output_shape": out_shape,
                    "parameters": module_params,
                    "flops": flops
                })
            return hook_fn

        for name, submodule in model.named_modules():
            if len(list(submodule.children())) == 0 and not isinstance(submodule, nn.Identity):
                hooks.append(submodule.register_forward_hook(create_hook(name)))

        dummy_input = torch.randn(*input_tensor_shape, device=target_device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
            for h in hooks:
                h.remove()

        latencies_ms: List[float] = []
        with torch.no_grad():
            for _ in range(benchmark_runs):
                if target_device.type == "cuda":
                    torch.cuda.synchronize()
                t_start = time.perf_counter_ns()
                _ = model(dummy_input)
                if target_device.type == "cuda":
                    torch.cuda.synchronize()
                t_end = time.perf_counter_ns()
                latencies_ms.append((t_end - t_start) / 1e6)

        mean_latency = float(np.mean(latencies_ms))
        std_latency = float(np.std(latencies_ms))
        p95_latency = float(np.percentile(latencies_ms, 95))
        min_latency = float(np.min(latencies_ms))
        max_latency = float(np.max(latencies_ms))
        fps = float(1000.0 / mean_latency) if mean_latency > 0 else 0.0

        model_config = model.get_architecture_config() if hasattr(model, "get_architecture_config") else {"model_class": model.__class__.__name__}
        inspection_result = {
            "model_config": model_config,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "non_trainable_parameters": non_trainable_parameters,
            "buffer_elements": buffer_elements,
            "model_memory_mb": round(model_memory_mb, 3),
            "total_flops": total_flops,
            "total_gflops": round(total_flops / 1e9, 4),
            "benchmark_device": str(target_device),
            "latency_mean_ms": round(mean_latency, 3),
            "latency_std_ms": round(std_latency, 3),
            "latency_p95_ms": round(p95_latency, 3),
            "latency_min_ms": round(min_latency, 3),
            "latency_max_ms": round(max_latency, 3),
            "throughput_fps": round(fps, 1),
            "layers": layer_records
        }
        return inspection_result

    @staticmethod
    def generate_markdown_report(inspection_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        cfg = inspection_data.get("model_config", {})
        arch_name = cfg.get("model_type", "GazeModel")
        backbone = cfg.get("backbone_name", cfg.get("backbone", "custom"))
        num_bins = cfg.get("num_bins", "N/A")
        total_p = inspection_data["total_parameters"]
        train_p = inspection_data["trainable_parameters"]
        mem_mb = inspection_data["model_memory_mb"]
        gflops = inspection_data["total_gflops"]
        lat_mean = inspection_data["latency_mean_ms"]
        lat_p95 = inspection_data["latency_p95_ms"]
        fps = inspection_data["throughput_fps"]
        dev = inspection_data["benchmark_device"]

        h1 = chr(35); h2 = chr(35) * 2
        md_lines = [
            f"{h1} Автоматический архитектурный отчет: {arch_name}",
            f"",
            f"{h2} 1. Сводные метрики вычислительной сложности",
            f"",
            f"| Метрика | Значение | Описание |",
            f"| :--- | :--- | :--- |",
            f"| **Архитектура** | `{arch_name}` | Идентификатор архитектуры |",
            f"| **Бэкбон** | `{backbone}` | Сверточный экстрактор признаков |",
            f"| **Число бинов (K)** | `{num_bins}` | Дискретизация углового пространства |",
            f"| **Всего параметров** | `{total_p:,}` | Суммарное число параметров |",
            f"| **Обучаемые параметры** | `{train_p:,}` | Параметры с градиентом |",
            f"| **Объем весов в памяти** | `{mem_mb} MB` | Память в формате float32 |",
            f"| **Вычислительная сложность** | `{gflops} GFLOPs` | Число операций с плавающей точкой |",
            f"| **Средняя задержка (Mean)** | `{lat_mean} ms` | Устройство: `{dev}` |",
            f"| **95-й перцентиль задержки (P95)** | `{lat_p95} ms` | Гарантированное время кадра |",
            f"| **Пропускная способность** | `{fps} FPS` | Кадров в секунду |",
            f"",
            f"{h2} 2. Математическая декомпозиция слоев",
            f"",
            r"1. **Извлечение признаков:** $\mathbf{F} = \operatorname{Backbone}(\mathbf{I}) \in \mathbb{R}^{C \times H \times W}$.",
            f"2. **Глобальное усреднение:** $\\mathbf{{f}} = \\operatorname{{AdaptiveAvgPool}}(\\mathbf{{F}}) \\in \\mathbb{{R}}^{{D}}$, где $D = {cfg.get('feature_dimension', 576)}$.",
            r"3. **Классификационные логиты тангажа и рыскания:**",
            r"   $$\mathbf{z}_{\theta} = \mathbf{W}_{\theta} \mathbf{f} + \mathbf{b}_{\theta} \in \mathbb{R}^K, \quad \mathbf{z}_{\phi} = \mathbf{W}_{\phi} \mathbf{f} + \mathbf{b}_{\phi} \in \mathbb{R}^K$$",
            r"4. **Распределение вероятностей (Softmax):**",
            r"   $$p_k = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}, \quad k \in \{1, \dots, K\}.$$",
            r"5. **Математическое ожидание углов (Регрессия):**",
            r"   $$\hat{\theta} = \sum_{k=1}^K p_k^\theta c_k, \quad \hat{\phi} = \sum_{k=1}^K p_k^\phi c_k.$$",
            r"6. **Единичный 3D-вектор направления взгляда:**",
            r"   $$\hat{\mathbf{g}} = \begin{bmatrix} -\cos\hat{\theta}\sin\hat{\phi} \\ -\sin\hat{\theta} \\ -\cos\hat{\theta}\cos\hat{\phi} \end{bmatrix}, \quad \|\hat{\mathbf{g}}\|_2 = 1.$$",
            f"",
            f"{h2} 3. Граф вычислительного конвейера (Mermaid)",
            f"",
            f"```mermaid",
            f"graph TD",
            f"    A[\"Входной кадр (B, 3, 224, 224)\"] --> B[\"Backbone Features\"]",
            f"    B --> C[\"AdaptiveAvgPool (B, {cfg.get('feature_dimension', 576)})\"]",
            f"    C --> D[\"Head Pitch Bins (B, {num_bins})\"]",
            f"    C --> E[\"Head Yaw Bins (B, {num_bins})\"]",
            f"    D --> F[\"Softmax Expectation -> Pitch theta\"]",
            f"    E --> G[\"Softmax Expectation -> Yaw phi\"]",
            f"    F --> H[\"3D Vector Transform -> g (B, 3)\"]",
            f"    G --> H",
            f"```",
            f"",
            f"{h2} 4. Послойный профиль архитектуры",
            f"",
            f"| Слой | Тип | Размерность входа | Размерность выхода | Параметры | FLOPs |",
            f"| :--- | :--- | :--- | :--- | :--- | :--- |"
        ]

        for layer in inspection_data.get("layers", [])[:35]:
            l_name = layer["name"]
            l_type = layer["type"]
            in_s = str(layer["input_shape"])
            out_s = str(layer["output_shape"])
            p_cnt = f"{layer['parameters']:,}"
            fl = f"{layer['flops']:,}"
            md_lines.append(f"| `{l_name}` | `{l_type}` | `{in_s}` | `{out_s}` | {p_cnt} | {fl} |")

        if len(inspection_data.get("layers", [])) > 35:
            md_lines.append(f"| ... | ... | ... | ... | ... | (Всего слоев: {len(inspection_data['layers'])}) |")

        report_content = "\n".join(md_lines)
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            metric_log(f"Architecture report written to: {os.path.abspath(output_path)}")
        return report_content

    @staticmethod
    def save_specification(inspection_data: Dict[str, Any], output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(inspection_data, f, indent=2, ensure_ascii=False)
        metric_log(f"Architecture spec written to: {os.path.abspath(output_path)}")
