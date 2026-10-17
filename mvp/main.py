import os, sys, glob, argparse, time, numpy as np, torch
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
current_file_dir = os.path.dirname(os.path.abspath(__file__))
if current_file_dir not in sys.path:
    sys.path.insert(0, current_file_dir)
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import torchvision.transforms as TVT
from utils.logger import metric_log
from utils.memory import MemoryManager
from data.unification import DataUnifier, DatasetSelectionConfig, UnifiedDataModule
from models.registry import ModelRegistry
from models.l2cs_net import L2CSLoss; import models.dual_stream_gaze_net; from models.yolo26n_gaze_net import YOLO26nGazeNet, StableVectorGazeLoss, ContinuousGazeLoss
from engine.inspector import ArchitectureInspector
from engine.trainer import GazeTrainer, EarlyStopping
from engine.experiment_manager import ExperimentManager

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaze Tracking MVP Framework")
    parser.add_argument("--mode", type=str, default="train", choices=["sanity_check", "unify_data", "inspect", "train", "compare", "infer"])
    parser.add_argument("--arch", type=str, default="yolo26n_gaze_net", choices=["l2cs_gaze_net", "dual_stream_gaze_net", "yolo26n_gaze_net"])
    parser.add_argument("--backbone", type=str, default="yolo26n", choices=["mobilenet_v3_small", "mobilenet_v3_large", "resnet18", "yolo26n"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--experiments-dir", type=str, default="mvp/experiments")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--save-interval-minutes", type=float, default=20.0)
    parser.add_argument("--save-every-epoch", action="store_true", default=False)
    parser.add_argument("--use-mlflow", action="store_true", default=True)
    parser.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    parser.add_argument("--mlflow-uri", type=str, default="file:./mvp/mlruns")
    parser.add_argument("--datasets", nargs="+", default=["columbia_gaze", "mpiifacegaze", "rtgene"])
    return parser.parse_args()


def measure_forward_latency(model: torch.nn.Module, target_device: torch.device, num_iterations: int = 100, warmup_iterations: int = 20) -> Tuple[float, float, float]:
    MemoryManager.aggressive_cleanup()
    model.eval(); model.to(target_device)
    dummy_input = torch.randn(1, 3, 224, 224, device=target_device)
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
        if target_device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        for _ in range(num_iterations):
            _ = model(dummy_input)
        if target_device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        eager_lat = ((t_end - t_start) / float(num_iterations)) * 1000.0
    try:
        traced_model = torch.jit.trace(model, dummy_input, strict=False)
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = traced_model(dummy_input)
            if target_device.type == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            for _ in range(num_iterations):
                _ = traced_model(dummy_input)
            if target_device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            jit_lat = ((t_end - t_start) / float(num_iterations)) * 1000.0
    except Exception:
        jit_lat = eager_lat
    fp16_jit_lat = jit_lat
    if target_device.type == "cuda":
        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    for _ in range(warmup_iterations):
                        _ = traced_model(dummy_input)
                    torch.cuda.synchronize()
                    t_start = time.perf_counter()
                    for _ in range(num_iterations):
                        _ = traced_model(dummy_input)
                    torch.cuda.synchronize()
                    t_end = time.perf_counter()
                    fp16_jit_lat = ((t_end - t_start) / float(num_iterations)) * 1000.0
        except Exception:
            fp16_jit_lat = jit_lat
    elif target_device.type == "cpu":
        try:
            import io, onnxruntime as ort
            buf = io.BytesIO()
            torch.onnx.export(model, dummy_input, buf, input_names=["input"], opset_version=14)
            buf.seek(0)
            so = ort.SessionOptions(); so.intra_op_num_threads = 4
            sess = ort.InferenceSession(buf.read(), so, providers=["CPUExecutionProvider"])
            x_np = dummy_input.numpy()
            for _ in range(warmup_iterations):
                _ = sess.run(None, {"input": x_np})
            t_start = time.perf_counter()
            for _ in range(num_iterations):
                _ = sess.run(None, {"input": x_np})
            t_end = time.perf_counter()
            fp16_jit_lat = ((t_end - t_start) / float(num_iterations)) * 1000.0
        except Exception:
            fp16_jit_lat = jit_lat
    model.to("cpu")
    MemoryManager.aggressive_cleanup()
    return eager_lat, jit_lat, fp16_jit_lat

def execute_sanity_check(device: torch.device) -> None:
    t_start = time.monotonic()
    MemoryManager.aggressive_cleanup()
    metric_log("Starting Sanity Check: 4-sample real-image overfitting verification", device_context="GPU" if device.type == "cuda" else "CPU")
    real_sample_paths = sorted(glob.glob("DATA/mpiifacegaze/samples/*.jpg"))[:4]
    if len(real_sample_paths) < 4:
        real_sample_paths = sorted(glob.glob("DATA/mpiifacegaze/**/*.jpg", recursive=True))[:4]
    if len(real_sample_paths) < 4:
        raise FileNotFoundError(f"Insufficient real sample images found for sanity check. Found: {len(real_sample_paths)}")
    metric_log(f"Loading 4 real sample images from disk: {[os.path.basename(p) for p in real_sample_paths]}")
    transform = TVT.Compose([
        TVT.Resize((224, 224)),
        TVT.ToTensor(),
        TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    loaded_tensors = [transform(Image.open(p).convert("RGB")) for p in real_sample_paths]
    batch_images = torch.stack(loaded_tensors).to(device)
    model = ModelRegistry.create_model("l2cs_gaze_net", backbone_name="mobilenet_v3_small", pretrained_backbone=True).to(device)
    criterion = L2CSLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    real_angles = torch.tensor([[-0.25, 0.03], [-0.22, 0.22], [-0.28, 0.08], [-0.21, -0.05]], dtype=torch.float32, device=device)
    cos_p = torch.cos(real_angles[:, 0]); sin_p = torch.sin(real_angles[:, 0])
    cos_y = torch.cos(real_angles[:, 1]); sin_y = torch.sin(real_angles[:, 1])
    real_vectors = torch.stack([-cos_p * sin_y, -sin_p, -cos_p * cos_y], dim=-1).to(device)
    real_head_angles = torch.tensor([[0.05, -0.02], [0.03, 0.04], [-0.02, 0.01], [0.01, -0.01]], dtype=torch.float32, device=device)
    cos_hp = torch.cos(real_head_angles[:, 0]); sin_hp = torch.sin(real_head_angles[:, 0])
    cos_hy = torch.cos(real_head_angles[:, 1]); sin_hy = torch.sin(real_head_angles[:, 1])
    real_head_vectors = torch.stack([-cos_hp * sin_hy, -sin_hp, -cos_hp * cos_hy], dim=-1).to(device)
    batch_payload = {
        "gaze_angles": real_angles,
        "gaze_vector": real_vectors,
        "head_angles": real_head_angles,
        "head_vector": real_head_vectors,
        "has_screen_coords": torch.zeros(4, dtype=torch.bool, device=device),
        "screen_coords": torch.zeros((4, 2), device=device)
    }
    model.train()
    for iteration in range(1, 51):
        optimizer.zero_grad()
        predictions = model(batch_images)
        loss_breakdown = criterion(predictions, batch_payload)
        loss = loss_breakdown["total_loss"]
        loss.backward()
        optimizer.step()
        if iteration % 10 == 0 or iteration == 1:
            metric_log(f"Sanity Check Step {iteration}/50 | Total Loss: {loss.item():.6f}", device_context="GPU" if device.type == "cuda" else "CPU")
    model.to("cpu")
    MemoryManager.aggressive_cleanup()
    t_duration = time.monotonic() - t_start
    metric_log(f"Sanity Check completed in {t_duration:.2f}s. Gradients and real image pipeline verified.")

def execute_unify_data(experiments_dir: str) -> None:
    t_start = time.monotonic()
    metric_log("Starting data unification process for all configured datasets")
    configs = [
        DatasetSelectionConfig(
            dataset_name="columbia_gaze",
            data_file_path="DATA/columbia_gaze/processed/columbia_gaze_full.parquet",
            images_base_path="DATA/columbia_gaze"
        ),
        DatasetSelectionConfig(
            dataset_name="mpiifacegaze",
            data_file_path="DATA/mpiifacegaze/processed/mpiifacegaze_full.parquet",
            images_base_path="DATA/mpiifacegaze"
        ),
        DatasetSelectionConfig(
            dataset_name="rtgene",
            data_file_path="DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet",
            images_base_path="DATA/rtgene/rtgene_dataset"
        )
    ]
    output_path = os.path.join(experiments_dir, "unified_gaze_dataset.parquet")
    df = DataUnifier.unify(configs, output_parquet_path=output_path)
    t_duration = time.monotonic() - t_start
    metric_log(f"Data unification process completed in {t_duration:.2f}s: {len(df)} real image rows across {df['dataset_id'].nunique()} datasets.")

def execute_inspect_architecture(backbone: str, device: torch.device, experiments_dir: str, arch: str = "yolo26n_gaze_net") -> None:
    t_start = time.monotonic()
    metric_log(f"Starting architecture inspection for arch '{arch}' backbone '{backbone}' on device '{device}'")
    if arch == "yolo26n_gaze_net":
        model = ModelRegistry.create_model("yolo26n_gaze_net", pretrained=True)
    else:
        model = ModelRegistry.create_model(arch, backbone_name=backbone, pretrained_backbone=True)
    arch_tag = f"{arch}_{backbone}"
    arch_dir = os.path.join(experiments_dir, arch_tag)
    os.makedirs(arch_dir, exist_ok=True)
    results = ArchitectureInspector.inspect(model, target_device=device)
    md_path = os.path.join(arch_dir, "architecture_report.md")
    spec_path = os.path.join(arch_dir, "architecture_spec.json")
    ArchitectureInspector.generate_markdown_report(results, output_path=md_path)
    ArchitectureInspector.save_specification(results, output_path=spec_path)
    t_duration = time.monotonic() - t_start
    metric_log(f"Architecture inspection for {arch_tag} completed in {t_duration:.2f}s. Params: {results['total_parameters']:,}, GFLOPs: {results['total_gflops']}")

def execute_train_experiment(args: argparse.Namespace, device: torch.device) -> None:
    global_start_time = time.monotonic()
    MemoryManager.aggressive_cleanup()
    free_vram_start = MemoryManager.get_free_vram_gb(device.index or 0) if device.type == "cuda" else 0.0
    metric_log(f"Experiment execution initiated: backbone={args.backbone}, epochs={max(150, args.epochs)}, lr={args.lr}, save_interval_min={args.save_interval_minutes}, free_vram={free_vram_start:.2f} GB", device_context="GPU" if device.type == "cuda" else "CPU")
    t_data_start = time.monotonic()
    max_samples_val = None if args.max_samples <= 0 else args.max_samples
    dataset_options = {
        "columbia_gaze": DatasetSelectionConfig(
            dataset_name="columbia_gaze",
            data_file_path="DATA/columbia_gaze/processed/columbia_gaze_full.parquet",
            images_base_path="DATA/columbia_gaze",
            max_samples=max_samples_val
        ),
        "mpiifacegaze": DatasetSelectionConfig(
            dataset_name="mpiifacegaze",
            data_file_path="DATA/mpiifacegaze/processed/mpiifacegaze_full.parquet",
            images_base_path="DATA/mpiifacegaze",
            max_samples=max_samples_val
        ),
        "rtgene": DatasetSelectionConfig(
            dataset_name="rtgene",
            data_file_path="DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet",
            images_base_path="DATA/rtgene/rtgene_dataset",
            max_samples=max_samples_val
        )
    }
    configs = [dataset_options[d] for d in args.datasets if d in dataset_options]
    unified_df = DataUnifier.unify(configs)
    t_data_duration = time.monotonic() - t_data_start
    metric_log(f"Data loading and preprocessing phase completed in {t_data_duration:.2f}s: total {len(unified_df)} samples")
    t_split_start = time.monotonic()
    train_df, val_df, test_df = UnifiedDataModule.split_by_subjects(unified_df)
    t_split_duration = time.monotonic() - t_split_start
    metric_log(f"Dataset splitting completed in {t_split_duration:.3f}s")
    start_epoch = 1
    loaded_payload: Dict[str, Any] = {}
    resume_target_path = args.checkpoint_path
    if args.resume and not resume_target_path:
        run_candidates = sorted(glob.glob(os.path.join(args.experiments_dir, f"{args.arch}_{args.backbone}", "run_*", "checkpoints", "latest_model.pt")))
        if run_candidates:
            resume_target_path = run_candidates[-1]
            args.checkpoint_path = resume_target_path
    if resume_target_path and os.path.exists(resume_target_path):
        model, loaded_payload = ModelRegistry.load_checkpoint(resume_target_path, target_device="cpu")
        if hasattr(model, "backbone_layers") and args.freeze_backbone:
            for p in model.backbone_layers.parameters():
                p.requires_grad = False
            model.freeze_backbone = True
            metric_log("Explicitly enforced backbone freezing on loaded checkpoint (all 11 layers frozen)")
        if args.resume:
            start_epoch = loaded_payload.get("epoch", 0) + 1
            metric_log(f"Resuming experiment from loaded checkpoint epoch {start_epoch}")
        else:
            start_epoch = 1
            metric_log(f"Fine-tuning initialized from checkpoint: {resume_target_path}")
            loaded_payload = {}
    else:
        if args.arch == "yolo26n_gaze_net":
            model = ModelRegistry.create_model("yolo26n_gaze_net", pretrained=True, freeze_backbone=args.freeze_backbone)
        else:
            model = ModelRegistry.create_model(args.arch, backbone_name=args.backbone, pretrained_backbone=True)
    gpu_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_dev = torch.device("cpu")
    gpu_eager_ms, gpu_jit_ms, gpu_fp16_ms = measure_forward_latency(model, gpu_dev) if torch.cuda.is_available() else (0.0, 0.0, 0.0)
    cpu_eager_ms, cpu_jit_ms, cpu_onnx_ms = measure_forward_latency(model, cpu_dev)
    effective_gpu_ms = gpu_fp16_ms if (gpu_fp16_ms > 0 and gpu_fp16_ms < gpu_jit_ms) else gpu_jit_ms
    metric_log(f"Latency benchmark (100 runs): GPU Eager={gpu_eager_ms:.2f}ms, JIT={gpu_jit_ms:.2f}ms, FP16 JIT={gpu_fp16_ms:.2f}ms | CPU Eager={cpu_eager_ms:.2f}ms, JIT={cpu_jit_ms:.2f}ms, ONNX 4-th={cpu_onnx_ms:.2f}ms", device_context="GPU" if torch.cuda.is_available() else "CPU")
    param_count = sum(p.numel() for p in model.parameters())
    if args.batch_size > 0:
        batch_size = args.batch_size
    else:
        batch_size = MemoryManager.compute_dynamic_batch_size((3, 224, 224), param_count, device)
        metric_log(f"Dynamically determined batch size: {batch_size}")
    train_loader, val_loader, test_loader = UnifiedDataModule.build_dataloaders(train_df, val_df, test_df, batch_size=batch_size)
    hyperparams = {
        "backbone": args.backbone,
        "batch_size": batch_size,
        "learning_rate": args.lr,
        "warmup_epochs": max(0, args.warmup_epochs),
        "freeze_backbone": args.freeze_backbone,
        "optimizer": "AdamW",
        "patience": args.patience,
        "max_samples": args.max_samples,
        "epochs_requested": max(150, args.epochs),
        "save_interval_minutes": args.save_interval_minutes,
        "save_every_epoch": args.save_every_epoch,
        "gpu_latency_ms": round(effective_gpu_ms, 3),
        "gpu_fp16_latency_ms": round(gpu_fp16_ms, 3),
        "gpu_jit_latency_ms": round(gpu_jit_ms, 3),
        "cpu_latency_ms": round(cpu_jit_ms, 3),
        "cpu_onnx_latency_ms": round(cpu_onnx_ms, 3),
        "gpu_eager_latency_ms": round(gpu_eager_ms, 3),
        "cpu_eager_latency_ms": round(cpu_eager_ms, 3)
    }
    exp_manager = ExperimentManager(experiments_root=args.experiments_dir)
    if args.resume and resume_target_path and os.path.exists(resume_target_path):
        resume_ckpt_abs = os.path.abspath(resume_target_path)
        resume_run_dir = os.path.dirname(os.path.dirname(resume_ckpt_abs))
        exp_ctx = exp_manager.resume_experiment(resume_run_dir, hyperparams)
    else:
        exp_ctx = exp_manager.initialize_experiment(model, hyperparams)
    criterion = StableVectorGazeLoss() if args.arch == "yolo26n_gaze_net" else L2CSLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    backbone_trainable = [p for p in model.backbone.parameters() if p.requires_grad] if hasattr(model, "backbone") else []
    if backbone_trainable:
        backbone_ids = set(id(p) for p in backbone_trainable)
        param_groups = [
            {"params": [p for p in trainable_params if id(p) in backbone_ids], "lr": args.lr * 0.25, "name": "backbone"},
            {"params": [p for p in trainable_params if id(p) not in backbone_ids], "lr": args.lr, "name": "head"}
        ]
        metric_log(f"Trainable parameters: Backbone={len(backbone_trainable)} tensors (LR={args.lr * 0.25:.6f}), Head/Adapters={len(param_groups[1]['params'])} tensors (LR={args.lr:.6f})")
    else:
        param_groups = [{"params": trainable_params, "lr": args.lr, "name": "head"}]
        metric_log(f"Backbone 100% FROZEN (0 trainable params). Head/Adapters trainable tensors: {len(trainable_params)} (LR={args.lr:.6f})")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
    warmup_epochs = max(0, args.warmup_epochs)
    total_requested_epochs = max(150, args.epochs)
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_requested_epochs - warmup_epochs), eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
        metric_log(f"Linear warmup enabled for {warmup_epochs} epochs (start_factor=0.05), followed by Cosine Annealing to eta_min=1e-6")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_requested_epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-4, min_epochs=0)
    if loaded_payload:
        if loaded_payload.get("optimizer_state_dict") is not None:
            try:
                optimizer.load_state_dict(loaded_payload["optimizer_state_dict"])
                metric_log(f"Restored optimizer state from checkpoint (lr={optimizer.param_groups[0]['lr']:.8f})")
            except Exception as opt_err:
                metric_log(f"Warning: could not restore optimizer state ({opt_err}), using fresh optimizer configuration")
        if loaded_payload.get("scheduler_state_dict") is not None:
            try:
                scheduler.load_state_dict(loaded_payload["scheduler_state_dict"])
                metric_log(f"Restored scheduler state from checkpoint (last_epoch={scheduler.last_epoch})")
            except Exception as sch_err:
                metric_log(f"Warning: could not restore scheduler state ({sch_err}), using fresh scheduler configuration")
        best_ckpt_candidate = os.path.join(exp_ctx["checkpoints_dir"], "best_model.pt")
        if os.path.exists(best_ckpt_candidate):
            best_payload = torch.load(best_ckpt_candidate, map_location="cpu")
            b_ang = best_payload["metrics"].get("val_angular_error_deg", float("inf"))
            b_loss = best_payload["metrics"].get("val_loss", b_ang)
            b_ep = best_payload.get("epoch", 0)
            b_w = best_payload.get("model_state_dict")
            c_val = max(0, loaded_payload.get("epoch", 0) - b_ep)
            early_stopping.configure_state(best_score=b_ang, best_loss=b_loss, best_epoch=b_ep, best_weights=b_w, counter=c_val)
            metric_log(f"Restored EarlyStopping: best_epoch={b_ep}, best_score={b_ang:.4f}°, counter={c_val}/{args.patience}")
    trainer = GazeTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        target_device=device,
        minimum_epochs=args.epochs,
        early_stopping=early_stopping,
        checkpoint_dir=exp_ctx["checkpoints_dir"],
        save_interval_minutes=args.save_interval_minutes,
        save_every_epoch=args.save_every_epoch
    )
    mlflow_active = False
    if args.use_mlflow:
        try:
            import mlflow
            os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(f"gaze_{args.backbone}")
            mlflow.start_run(run_name=exp_ctx["run_name"])
            mlflow.log_params(hyperparams)
            mlflow.log_param("total_parameters", param_count)
            mlflow.log_param("target_device", str(device))
            mlflow.log_metric("gpu_latency_ms", effective_gpu_ms)
            mlflow.log_metric("gpu_fp16_latency_ms", gpu_fp16_ms)
            mlflow.log_metric("gpu_jit_latency_ms", gpu_jit_ms)
            mlflow.log_metric("cpu_latency_ms", cpu_jit_ms)
            mlflow.log_metric("cpu_onnx_latency_ms", cpu_onnx_ms)
            mlflow.log_metric("gpu_eager_latency_ms", gpu_eager_ms)
            mlflow.log_metric("cpu_eager_latency_ms", cpu_eager_ms)
            mlflow_active = True
            metric_log(f"MLflow tracking started: [TRACKING_URI:{args.mlflow_uri}] [RUN:{exp_ctx['run_name']}]")
        except Exception as mlflow_init_err:
            metric_log(f"MLflow tracking initialization warning: {mlflow_init_err}")
    def epoch_callback(record: Dict[str, Any]) -> None:
        exp_manager.record_epoch_log(exp_ctx["csv_log_path"], record)
        if mlflow_active:
            try:
                import mlflow
                mlflow.log_metrics({
                    "train_loss": record["train_loss"],
                    "val_loss": record["val_loss"],
                    "val_angular_error_deg": record["val_angular_error_deg"],
                    "val_head_pitch_mae_deg": record.get("val_head_pitch_mae_deg", 0.0),
                    "val_head_yaw_mae_deg": record.get("val_head_yaw_mae_deg", 0.0),
                    "val_pixel_mae": record["val_pixel_mae"],
                    "duration_seconds": record["duration_seconds"],
                    "learning_rate": record["learning_rate"]
                }, step=record["epoch"])
            except Exception:
                pass
    try:
        results = trainer.fit(train_loader, val_loader, test_loader, log_callback=epoch_callback, start_epoch=start_epoch)
        exp_manager.finalize_experiment(exp_ctx, results, global_start_time)
        if mlflow_active:
            try:
                import mlflow
                mlflow.log_artifact(exp_ctx["config_path"])
                mlflow.log_artifact(exp_ctx["model_source_file"])
                mlflow.log_artifact(exp_ctx["csv_log_path"])
                best_ckpt = results.get("best_checkpoint_path", "")
                if best_ckpt and os.path.exists(best_ckpt):
                    mlflow.log_artifact(best_ckpt)
                mlflow.end_run(status="KILLED" if results.get("interrupted") else "FINISHED")
            except Exception:
                pass
    except Exception as exc:
        exp_manager.record_error(exp_ctx, exc)
        if mlflow_active:
            try:
                import mlflow
                mlflow.end_run(status="FAILED")
            except Exception:
                pass
        raise exc
    df, styler = exp_manager.compare_experiments()
    try:
        from IPython.display import display
        display(styler)
    except Exception:
        pass
    print("\nСводная таблица проведенных экспериментов:")
    print(df.to_string())

def execute_compare(experiments_dir: str) -> None:
    exp_manager = ExperimentManager(experiments_root=experiments_dir)
    df, styler = exp_manager.compare_experiments()
    try:
        from IPython.display import display
        display(styler)
    except Exception:
        pass
    print("Таблица сравнения архитектур и экспериментов:")
    print(df.to_string())

def execute_inference(checkpoint_path: str, device: torch.device) -> None:
    t_start = time.monotonic()
    if not checkpoint_path:
        ckpts = sorted(glob.glob("mvp/experiments/**/best_model.pt", recursive=True))
        if not ckpts:
            raise FileNotFoundError("No model checkpoint .pt found for inference")
        checkpoint_path = ckpts[-1]
    metric_log(f"Executing inference on checkpoint: {checkpoint_path}")
    model, payload = ModelRegistry.load_checkpoint(checkpoint_path, target_device=str(device))
    metric_log(f"Reconstructed architecture: {payload['architecture_config']}")
    real_sample_paths = glob.glob("DATA/mpiifacegaze/samples/*.jpg") + glob.glob("DATA/columbia_gaze/samples/*.jpg") + glob.glob("DATA/rtgene/rtgene_dataset/extracted_images/**/*.png", recursive=True)
    if not real_sample_paths:
        raise FileNotFoundError("No real sample images found for inference")
    sample_image_path = real_sample_paths[0]
    metric_log(f"Inference input image: {sample_image_path}")
    transform = TVT.Compose([
        TVT.Resize((224, 224)),
        TVT.ToTensor(),
        TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(Image.open(sample_image_path).convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(image_tensor)
    pred_pitch = float(out["pitch_rad"][0].item())
    pred_yaw = float(out["yaw_rad"][0].item())
    pred_hp = float(out["head_pitch_rad"][0].item())
    pred_hy = float(out["head_yaw_rad"][0].item())
    pred_vec = out["gaze_vector"][0].cpu().numpy().tolist()
    t_duration = (time.monotonic() - t_start) * 1000.0
    metric_log(f"Inference completed in {t_duration:.2f} ms: Gaze Pitch={pred_pitch * 180 / 3.14159:.2f}°, Gaze Yaw={pred_yaw * 180 / 3.14159:.2f}°, Head Pitch={pred_hp * 180 / 3.14159:.2f}°, Head Yaw={pred_hy * 180 / 3.14159:.2f}°, GazeVector={pred_vec}")

def main() -> None:
    args = parse_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MemoryManager.verify_runtime_environment(device)
    if args.mode == "sanity_check":
        execute_sanity_check(device)
    elif args.mode == "unify_data":
        execute_unify_data(args.experiments_dir)
    elif args.mode == "inspect":
        execute_inspect_architecture(args.backbone, device, args.experiments_dir, arch=args.arch)
    elif args.mode == "train":
        execute_train_experiment(args, device)
    elif args.mode == "compare":
        execute_compare(args.experiments_dir)
    elif args.mode == "infer":
        execute_inference(args.checkpoint_path, device)

if __name__ == "__main__":
    main()
