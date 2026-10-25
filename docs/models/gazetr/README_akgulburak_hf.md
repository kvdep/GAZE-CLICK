---
license: cc-by-nc-sa-4.0
tags:
  - gaze-estimation
  - computer-vision
  - fairness
  - bias-mitigation
datasets:
  - Gaze360
  - GazeCapture
---

# Gaze Estimation Diversity — Model Weights

Checkpoints for the models used in *"Investigating Bias and Fairness in
Appearance-based Gaze Estimation"* (Akgül, Şahin & Kalkan, 2026).

Full code, training scripts, annotations, and documentation are on GitHub:
**[akgulburak/gaze-estimation-fairness](https://github.com/akgulburak/gaze-estimation-fairness)**

## What's in this repo

Checkpoints for **5 architectures** (CrossGaze, GazeTR, L2CS-Net, MCGaze,
PureGaze), each trained under **4 conditions**:

- Baseline (no bias mitigation)
- Oversampling
- Resampling
- Loss reweighting

Each mitigation condition was trained separately targeting the **ethnicity**
attribute and the **gender** attribute, all on the Gaze360 dataset.

## File Naming

We repeated each run 5 times to account for training variance. The _1-_5 suffix in each filename indicates the repetition index - these are independent runs of the same configuration, not different models.

## Usage

See the [GitHub repo](https://github.com/akgulburak/gaze-estimation-fairness) for training/inference code.

## License

This project is licensed under CC BY-NC-SA 4.0 - see LICENSE for details.

## Citation

```bibtex
@inproceedings{akgul2026biasingaze,
  title     = {Investigating Bias and Fairness in Appearance-based Gaze Estimation},
  author    = {Akg{\"u}l, Burak and {\c{S}}ahin, Erol and Kalkan, Sinan},
  booktitle = {2026 IEEE 20th International Conference on Automatic Face and Gesture Recognition (FG)},
  year      = {2026},
  address   = {Kyoto, Japan},
  publisher = {IEEE},
  doi       = {10.1109/FG67764.2026.11557007}
}
```