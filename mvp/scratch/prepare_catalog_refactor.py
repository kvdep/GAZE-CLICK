import re

file_path = "docs/datasets/catalog.md"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Let's inspect where we can place:
# 1. New Section: Исключенные и нерелевантные датасеты (Gaze-Following, Saliency, Съемка дороги)
# 2. Add HungSteven/VRGaze and MetaHuman Gaze
# 3. Update HybridGaze Kaggle link to https://www.kaggle.com/datasets/michachwesiuk/hybridgaze
# 4. Update SynthesEyes link to https://doi.org/10.17863/CAM.57942
# 5. Clean up the table in Section 2
