import os, sys, cv2, torch, importlib.util

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("Cannot read camera")
    sys.exit(1)

d = cv2.FaceDetectorYN.create("mvp/models/weights/face_detection_yunet_2023mar.onnx", "", (frame.shape[1], frame.shape[0]))
_, faces = d.detect(frame)
if faces is None or len(faces) == 0:
    print("No face detected")
    sys.exit(1)

fx, fy, fw, fh = faces[0][:4]
s = int(1.3 * max(fw, fh))
cx, cy = int(fx + fw / 2), int(fy + fh / 2)
crop = frame[max(0, cy - s // 2):min(frame.shape[0], cy + s // 2), max(0, cx - s // 2):min(frame.shape[1], cx + s // 2)]
crop = cv2.resize(crop, (224, 224))
rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
tin = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
tin = (tin - mean) / std

from models.registry import ModelRegistry
m_mob, _ = ModelRegistry.load_checkpoint("mvp/experiments/l2cs_gaze_net_mobilenet_v3_large/run_008_20261006_001852/checkpoints/best_model.pt", target_device="cpu")
m_mob.eval()
with torch.no_grad():
    out_mob = m_mob(tin)

spec = importlib.util.spec_from_file_location("m_res", "mvp/experiments/l2cs_gaze_net_resnet18/run_001_20261004_163937/model_source.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
m_res = mod.L2CSGazeNet(backbone_name="resnet18")
d_res = torch.load("mvp/experiments/l2cs_gaze_net_resnet18/run_001_20261004_163937/checkpoints/best_model.pt", map_location="cpu", weights_only=False)
m_res.load_state_dict(d_res["model_state_dict"])
m_res.eval()
with torch.no_grad():
    out_res = m_res(tin)

p_mob = float(out_mob["pitch_rad"].item() * 57.2957795)
y_mob = float(out_mob["yaw_rad"].item() * 57.2957795)
p_res = float(out_res["pitch_rad"].item() * 57.2957795)
y_res = float(out_res["yaw_rad"].item() * 57.2957795)

print(f"MobileNetV3: pitch={p_mob:.2f} deg, yaw={y_mob:.2f} deg")
print(f"ResNet18:    pitch={p_res:.2f} deg, yaw={y_res:.2f} deg")
