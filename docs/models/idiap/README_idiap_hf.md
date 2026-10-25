---
license: cc-by-nc-4.0
---

# resnet18-gaze360

Gaze following and social gaze prediction are fundamental tasks providing insights into human communication behaviors, intent, and social interactions. Most previous approaches addressed these tasks separately, either by designing highly specialized social gaze models that do not generalize to other social gaze tasks or by considering social gaze inference as an ad-hoc post-processing of the gaze following task. Furthermore, the vast majority of gaze following approaches have proposed models that can handle only one person at a time and are static, therefore failing to take advantage of social interactions and temporal dynamics. In this paper, we address these limitations and introduce a novel framework to jointly predict the gaze target and social gaze label for all people in the scene. It comprises a temporal, transformer-based architecture that, in addition to frame tokens, handles person-specific tokens capturing the gaze information related to each individual. We demonstrate that our model can address and benefit from training on all tasks jointly, achieving state-of-the-art results for multi-person gaze following and social gaze prediction.

## Overview

* **Training**: resnet18-gaze360 was trained on [Gaze360](https://github.com/erkil1452/gaze360/)
* **Backbone**: resnet18-gaze360 is adapted from [ResNet-18](https://huggingface.co/microsoft/resnet-18)
  * License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.html)
* **Parameters**: 11M
* **Task**: MTGS performs multi-person gaze following and social gaze prediction in images and videos. Given an image or a video frame with multiple people, the model predicts where each person is looking in the scene (gaze following) and infers pair-wise social gaze interactions among individuals (social gaze prediction). Specifically, we tackle three social gaze prediction tasks:
  1. Looking at Heads (LAH): whether a person is looking at another person's head.
  2. Looking at Each Other (LAEO): whether two people are looking at each other.
  3. Shared Attention (SA): whether two people are looking at the same target in the scene.
* **Framework**: PyTorch Lightning

## License

Copyright (c) 2026 Idiap Research Institute

**CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International)
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

## Minimal code to instantiate the model and perform inference:

Refer to instructions in the official repository: [https://github.com/idiap/MTGS](https://github.com/idiap/MTGS)

## Model performance

This is a component of the MTGS model.

## Citation

If you use these model, please cite the following publication:

```bibtex
@article{gupta2024mtgs,
title={MTGS: A Novel Framework for Multi-Person Temporal Gaze Following and Social Gaze Prediction},
author={Gupta, Anshul and Tafasca, Samy and Farkhondeh, Arya and Vuillecard, Pierre and Odobez, Jean-Marc},
journal={Advances in Neural Information Processing Systems},
volume={37},
pages={15646--15673},
year={2024}
}
```
