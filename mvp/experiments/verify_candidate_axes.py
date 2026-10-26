import os, sys, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load session 20261013_182824
vid_path = "mvp/benchmarks/benchmark_test_20261013_182824.mp4"
csv_path = "mvp/benchmarks/benchmark_test_20261013_182824.csv"
cache_path = "mvp/experiments/_cache_face_20261013_182824.npz"

df = pd.read_csv(csv_path)
cache = np.load(cache_path)
bboxes = cache["bboxes"]

# Find indices for Left, Right, Top, Bottom targets
# Left: target_x < 500, target_y approx 900
# Right: target_x > 2500, target_y approx 900
# Top: target_x approx 1440, target_y < 400
# Bottom: target_x approx 1440, target_y > 1500
idx_left = df[(df['target_gt_x'] < 500) & (df['target_gt_y'].between(800, 1000))].index[0]
idx_right = df[(df['target_gt_x'] > 2500) & (df['target_gt_y'].between(800, 1000))].index[0]
idx_top = df[(df['target_gt_x'].between(1300, 1550)) & (df['target_gt_y'] < 400)].index[0]
idx_bottom = df[(df['target_gt_x'].between(1300, 1550)) & (df['target_gt_y'] > 1500)].index[0]

indices = {"Left": idx_left, "Right": idx_right, "Top": idx_top, "Bottom": idx_bottom}
print("Selected test frame indices:")
for name, idx in indices.items():
    print(f"  {name:<6}: frame {idx}, target=({df.loc[idx, 'target_gt_x']:.1f}, {df.loc[idx, 'target_gt_y']:.1f})")

# Read those frames
cap = cv2.VideoCapture(vid_path)
frames = {}
for i in range(max(indices.values()) + 1):
    ret, frame = cap.read()
    if not ret: break
    for name, idx in indices.items():
        if i == idx:
            frames[name] = frame
cap.release()

def extract_face_crop(frame, bbox, size=224):
    h, w = frame.shape[:2]
    fx, fy, fw, fh = bbox
    cx = int(fx + fw / 2.0); cy = int(fy + fh / 2.0)
    box_side = int(max(fw, fh) * 1.10)
    half_side = box_side // 2
    cx1 = max(0, cx - half_side); cy1 = max(0, cy - half_side)
    cx2 = min(w, cx + half_side); cy2 = min(h, cy + half_side)
    crop = frame[cy1:cy2, cx1:cx2]
    return cv2.resize(crop, (size, size))

# 1. Models definition
# Model A: ptgaze ETH-XGaze ResNet-18
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import torchvision.models as models

st_eth = load_file(hf_hub_download("hysts/ptgaze-eth-xgaze-resnet18", "model.safetensors"))
net_eth = models.resnet18(weights=None)
net_eth.fc = nn.Linear(512, 2)
net_eth.load_state_dict(st_eth)
net_eth.to(device).eval()

# Model B: UniGaze ViT-Base
import timm
st_uni = load_file(hf_hub_download("UniGaze/UniGaze-models", "unigaze_b16_joint.safetensors"))
vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
vit.load_state_dict({k.replace('vit.', ''): v for k, v in st_uni.items() if k.startswith('vit.')}, strict=False)
gaze_fc = nn.Linear(768, 2)
gaze_fc.weight.data.copy_(st_uni['gaze_fc.weight'])
gaze_fc.bias.data.copy_(st_uni['gaze_fc.bias'])
class UniGazeModel(nn.Module):
    def __init__(self, vit, gaze_fc):
        super().__init__()
        self.vit = vit
        self.gaze_fc = gaze_fc
    def forward(self, x):
        return self.gaze_fc(self.vit(x))
net_uni = UniGazeModel(vit, gaze_fc).to(device).eval()

# Model C: Yakhyo ResNet-50
from torchvision.models.resnet import Bottleneck
class GazeResNet50(models.ResNet):
    def __init__(self, num_bins=90):
        super().__init__(Bottleneck, [3, 4, 6, 3])
        self.fc_pitch = nn.Linear(2048, num_bins)
        self.fc_yaw = nn.Linear(2048, num_bins)
        del self.fc
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc_yaw(x), self.fc_pitch(x)
net_yak50 = GazeResNet50(num_bins=90)
net_yak50.load_state_dict(torch.load("mvp/models/weights/yakhyo_resnet50.pt", map_location="cpu"))
net_yak50.to(device).eval()

# Normalization constants
mean_imagenet = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std_imagenet = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

print("\n" + "="*80)
print("EVALUATING DIRECTIONAL RESPONSES")
print("="*80)

# Evaluate ptgaze ETH-XGaze
print("\n--- 1. ptgaze ETH-XGaze ResNet-18 (RGB ImageNet) ---")
for name in ["Left", "Right", "Top", "Bottom"]:
    crop = extract_face_crop(frames[name], bboxes[indices[name]])
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    t = (t - mean_imagenet) / std_imagenet
    with torch.no_grad():
        out = net_eth(t)[0].cpu().numpy()
    deg0 = math.degrees(out[0])
    deg1 = math.degrees(out[1])
    print(f"  {name:<6}: out[0]={deg0:+.2f}°, out[1]={deg1:+.2f}° (raw: [{out[0]:+.4f}, {out[1]:+.4f}])")

# Evaluate UniGaze
print("\n--- 2. UniGaze ViT-Base (RGB ImageNet) ---")
for name in ["Left", "Right", "Top", "Bottom"]:
    crop = extract_face_crop(frames[name], bboxes[indices[name]])
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    t = (t - mean_imagenet) / std_imagenet
    with torch.no_grad():
        out = net_uni(t)[0].cpu().numpy()
    deg0 = math.degrees(out[0])
    deg1 = math.degrees(out[1])
    print(f"  {name:<6}: out[0]={deg0:+.2f}°, out[1]={deg1:+.2f}° (raw: [{out[0]:+.4f}, {out[1]:+.4f}])")

# Evaluate ptgaze MPIIFaceGaze
print("\n--- 4. ptgaze MPIIFaceGaze ResNet-Simple (BGR ImageNet) ---")
class ResNetSimpleMPII(nn.Module):
    def __init__(self):
        super().__init__()
        r = models.resnet18(weights=None)
        self.conv1 = r.conv1; self.bn1 = r.bn1; self.relu = r.relu; self.maxpool = r.maxpool
        self.layer1 = r.layer1; self.layer2 = r.layer2; self.layer3 = r.layer3
        self.conv = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(50176, 2)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        y = F.relu(self.conv(x))
        x = x * y
        x = x.view(x.size(0), -1)
        return self.fc(x)

st_mpii = load_file(hf_hub_download("hysts/ptgaze-mpiifacegaze-resnet-simple", "model.safetensors"))
# Remap keys from "feature_extractor.conv1.weight" -> "conv1.weight"
remap_st = {}
for k, v in st_mpii.items():
    if k.startswith("feature_extractor."):
        remap_st[k.replace("feature_extractor.", "")] = v
    else:
        remap_st[k] = v
net_mpii = ResNetSimpleMPII()
net_mpii.load_state_dict(remap_st)
net_mpii.to(device).eval()

for name in ["Left", "Right", "Top", "Bottom"]:
    crop = extract_face_crop(frames[name], bboxes[indices[name]])
    # MPIIFaceGaze expects BGR channel order with ImageNet mean/std
    t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    t = (t - mean_imagenet) / std_imagenet
    with torch.no_grad():
        out = net_mpii(t)[0].cpu().numpy()
    deg0 = math.degrees(out[0])
    deg1 = math.degrees(out[1])
    print(f"  {name:<6}: out[0]={deg0:+.2f}°, out[1]={deg1:+.2f}° (raw: [{out[0]:+.4f}, {out[1]:+.4f}])")
