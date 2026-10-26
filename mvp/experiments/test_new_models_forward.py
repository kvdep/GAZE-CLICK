import os, sys, math, time
import numpy as np
import torch
import torch.nn as nn
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {device}")

# 1. Test ptgaze ETH-XGaze ResNet-18
def test_ptgaze_eth():
    print("\n--- 1. Testing ptgaze ETH-XGaze ResNet-18 ---")
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    import torchvision.models as models

    weight_path = hf_hub_download("hysts/ptgaze-eth-xgaze-resnet18", "model.safetensors")
    st = load_file(weight_path)

    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, 2)
    net.load_state_dict(st)
    net.to(device).eval()

    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = net(dummy)
    print("ptgaze ETH-XGaze output shape:", out.shape, "val:", out.cpu().numpy())
    return net

# 2. Test UniGaze ViT-Base
def test_unigaze():
    print("\n--- 2. Testing UniGaze ViT-Base ---")
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    import timm

    weight_path = hf_hub_download("UniGaze/UniGaze-models", "unigaze_b16_joint.safetensors")
    st = load_file(weight_path)

    vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    vit_weights = {k.replace('vit.', ''): v for k, v in st.items() if k.startswith('vit.')}
    vit.load_state_dict(vit_weights, strict=False)

    gaze_fc = nn.Linear(768, 2)
    gaze_fc.weight.data.copy_(st['gaze_fc.weight'])
    gaze_fc.bias.data.copy_(st['gaze_fc.bias'])

    class UniGazeModel(nn.Module):
        def __init__(self, vit, gaze_fc):
            super().__init__()
            self.vit = vit
            self.gaze_fc = gaze_fc
        def forward(self, x):
            feats = self.vit(x)
            return self.gaze_fc(feats)

    model = UniGazeModel(vit, gaze_fc).to(device).eval()

    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
    print("UniGaze output shape:", out.shape, "val:", out.cpu().numpy())
    return model

# 3. Test Yakhyo ResNet-50
def test_yakhyo_r50():
    print("\n--- 3. Testing Yakhyo ResNet-50 ---")
    import torchvision.models as models
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
            pitch = self.fc_pitch(x)
            yaw = self.fc_yaw(x)
            return yaw, pitch

    model = GazeResNet50(num_bins=90)
    sd = torch.load("mvp/models/weights/yakhyo_resnet50.pt", map_location="cpu")
    model.load_state_dict(sd)
    model.to(device).eval()

    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        yaw, pitch = model(dummy)
    print("Yakhyo R50 output shapes:", yaw.shape, pitch.shape)
    return model

# 4. Test Yakhyo ResNet-18
def test_yakhyo_r18():
    print("\n--- 4. Testing Yakhyo ResNet-18 ---")
    import torchvision.models as models
    from torchvision.models.resnet import BasicBlock

    class GazeResNet18(models.ResNet):
        def __init__(self, num_bins=90):
            super().__init__(BasicBlock, [2, 2, 2, 2])
            self.fc_pitch = nn.Linear(512, num_bins)
            self.fc_yaw = nn.Linear(512, num_bins)
            del self.fc
        def forward(self, x):
            x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
            x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
            x = self.avgpool(x).flatten(1)
            pitch = self.fc_pitch(x)
            yaw = self.fc_yaw(x)
            return yaw, pitch

    model = GazeResNet18(num_bins=90)
    sd = torch.load("mvp/models/weights/yakhyo_resnet18.pt", map_location="cpu")
    model.load_state_dict(sd)
    model.to(device).eval()

    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        yaw, pitch = model(dummy)
    print("Yakhyo R18 output shapes:", yaw.shape, pitch.shape)
    return model

# 5. Test ptgaze MPIIFaceGaze ResNet-Simple
def test_ptgaze_mpii():
    print("\n--- 5. Testing ptgaze MPIIFaceGaze ResNet-Simple ---")
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    import torchvision.models as models

    # ptgaze resnet_simple takes first 3 layers and a linear(50176, 2)
    class ResNetSimple(nn.Module):
        def __init__(self):
            super().__init__()
            r = models.resnet18(weights=None)
            self.conv1 = r.conv1; self.bn1 = r.bn1; self.relu = r.relu; self.maxpool = r.maxpool
            self.layer1 = r.layer1; self.layer2 = r.layer2; self.layer3 = r.layer3; self.layer4 = r.layer4
            self.fc = nn.Linear(50176, 2)
        def forward(self, x):
            x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
            x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
            x = x.flatten(1)
            return self.fc(x)

    weight_path = hf_hub_download("hysts/ptgaze-mpiifacegaze-resnet-simple", "model.safetensors")
    st = load_file(weight_path)
    model = ResNetSimple()
    model.load_state_dict(st)
    model.to(device).eval()

    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
    print("ptgaze MPIIFaceGaze output shape:", out.shape, "val:", out.cpu().numpy())
    return model

if __name__ == "__main__":
    test_ptgaze_eth()
    test_unigaze()
    test_yakhyo_r50()
    test_yakhyo_r18()
    test_ptgaze_mpii()
    print("\nAll 5 new candidate models successfully tested and validated on forward pass!")
