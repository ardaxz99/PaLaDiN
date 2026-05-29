# PALADIN

<div align="center">

# **PALADIN**

### **Prompt-Aligned Localization and Anomaly Detection with DINOv3**

**Accepted to CVPR 2026 Workshop on Vision-based Industrial Anomaly Detection — VAND**

[![CVPR 2026 Workshop](https://img.shields.io/badge/CVPR%202026-Workshop-blue)](https://openaccess.thecvf.com/content/CVPR2026W/VAND/html/Basaran_PALADIN_Prompt-Aligned_Localization_and_Anomaly_Detection_with_DINOv3_CVPRW_2026_paper.html)
[![Paper](https://img.shields.io/badge/Paper-CVF%20Open%20Access-red)](https://openaccess.thecvf.com/content/CVPR2026W/VAND/html/Basaran_PALADIN_Prompt-Aligned_Localization_and_Anomaly_Detection_with_DINOv3_CVPRW_2026_paper.html)
[![Datasets](https://img.shields.io/badge/Datasets-MVTec%20%7C%20VisA-green)](#prepare-the-data-folders)

</div>

---

## Overview

**PALADIN** is a prompt-based industrial anomaly detection framework designed for anomaly localization and image-level anomaly detection under the normal-only training setting.

The project combines:

* a **DINOv3 ViT-L/16** visual backbone
* **OpenCLIP** text embeddings
* **learnable text prompts**
* **Cross-Modal Alignment Adapters** for aligning visual patch features with text prototypes
* **guided synthetic anomaly generation** for training without real anomaly labels
* optional **top-k anomaly scoring** for evaluation

This repository currently supports:

* **MVTec-AD**
* **VisA**

---

## Abstract

Industrial anomaly detection requires identifying subtle and previously unseen defects when real anomaly examples are scarce. PALADIN addresses this setting by connecting strong self-supervised visual representations from DINOv3 with language-guided anomaly reasoning from CLIP-style text embeddings.

The method keeps the visual and text backbones frozen, while lightweight alignment modules project DINO patch features into the text embedding space. Dense anomaly maps are then produced by comparing patch-level visual features against prompt-conditioned normal and abnormal text prototypes. Training uses only normal images from the target dataset, together with guided synthetic anomalies that provide pixel-level supervision without requiring real anomaly labels.

PALADIN also supports few-shot adaptation through a non-parametric memory bank built from clean support samples. Experiments on MVTec-AD and VisA show that aligning self-supervised visual features with prompt-based language representations is an effective direction for scalable industrial anomaly detection and localization.

---

## Paper

**PALADIN: Prompt-Aligned Localization and Anomaly Detection with DINOv3**
Arda Basaran
*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2026*

* [Paper page](https://openaccess.thecvf.com/content/CVPR2026W/VAND/html/Basaran_PALADIN_Prompt-Aligned_Localization_and_Anomaly_Detection_with_DINOv3_CVPRW_2026_paper.html)
* [PDF](https://openaccess.thecvf.com/content/CVPR2026W/VAND/papers/Basaran_PALADIN_Prompt-Aligned_Localization_and_Anomaly_Detection_with_DINOv3_CVPRW_2026_paper.pdf)
* [Supplementary material](https://openaccess.thecvf.com/content/CVPR2026W/VAND/supplemental/Basaran_PALADIN_Prompt-Aligned_Localization_CVPRW_2026_supplemental.pdf)

---

## Setup

### 1. Create an Environment

You need a Python environment with the packages from [requirements.txt](/Users/abasaran/Desktop/PhD/PaLaDiN/requirements.txt:1), a working PyTorch + CUDA setup, and `deepspeed`. A simple setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The shell scripts automatically activate `./.venv` when it exists. To use a different environment, set `PALADIN_VENV`:

```bash
PALADIN_VENV=/path/to/env bash ./scripts/train_mvtec.sh
```

If no virtual environment is found, the scripts continue with the current shell environment.

### 2. Clone DINOv3 Into the Project Root

```bash
git clone https://github.com/facebookresearch/dinov3.git ./dinov3
```

The model code resolves this repo automatically from the project root.

### 3. Download the DINOv3 Checkpoint

Place this file in the project root:

```bash
./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

This is the checkpoint the model currently loads by default.

### 4. Prepare the Data Folders

Download the datasets:

* [MVTec-AD](https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz)
* [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)

The shell scripts currently expect these dataset roots:

```text
data/
|-- mvtec/
`-- visa/

fg_masks/
|-- mvtec_fg/
`-- visa_fg/
```

`mvtec` and `visa` are the actual datasets.

`mvtec_fg` and `visa_fg` are required for guided synthetic anomaly generation during training. Training will fail if these folders are missing.

You can download the foreground masks here:

* [fg_masks download](https://drive.google.com/drive/folders/1ohepx-HkTAf6CkAGVDKm7ZH-0-_0cwk-?usp=sharing)

To use dataset roots outside `./data`, set:

```bash
PALADIN_MVTEC_PATH=/path/to/mvtec
PALADIN_VISA_PATH=/path/to/visa
```

### 5. Expected Dataset Layout

MVTec should look like this:

```text
data/mvtec/
|-- bottle/
|   |-- ground_truth/
|   |-- test/
|   `-- train/
|-- cable/
|-- capsule/
`-- ...
```

VisA should look like this:

```text
data/visa/
|-- split_csv/
|   `-- 1cls.csv
|-- candle/
|   |-- Data/
|   |   `-- Images/
|   `-- Masks/
|-- capsules/
`-- ...
```

The guided-mask folders are expected under the repo-level `fg_masks/` directory:

```text
fg_masks/
|-- mvtec_fg/
`-- visa_fg/
```

---

## Training

### Train on MVTec

```bash
bash ./scripts/train_mvtec.sh
```

This script currently runs:

* `deepspeed`
* `code/train_mvtec.py`
* dataset path: `./data/mvtec`
* output path: `./code/mvtec_paladin/train_mvtec/`

If your local paths differ, edit [scripts/train_mvtec.sh](/Users/abasaran/Desktop/PhD/PaLaDiN/scripts/train_mvtec.sh:1).

To use a custom MVTec path:

```bash
PALADIN_MVTEC_PATH=/path/to/mvtec bash ./scripts/train_mvtec.sh
```

### Train on VisA

```bash
bash ./scripts/train_visa.sh
```

This script currently runs:

* `deepspeed`
* `code/train_visa.py`
* dataset path: `./data/visa`
* output path: `./code/visa_paladin/train_visa/`

If your local paths differ, edit [scripts/train_visa.sh](/Users/abasaran/Desktop/PhD/PaLaDiN/scripts/train_visa.sh:1).

To use a custom VisA path:

```bash
PALADIN_VISA_PATH=/path/to/visa bash ./scripts/train_visa.sh
```

---

## Evaluation

### Evaluate on MVTec

```bash
bash ./scripts/test_mvtec.sh paladin
```

For the top-k scoring variant:

```bash
bash ./scripts/test_mvtec.sh paladintopk
```

This script currently expects the checkpoint at:

```text
./code/mvtec_paladin/train_mvtec/epoch_10/pytorch_model.pt
```

If your checkpoint is elsewhere, edit [scripts/test_mvtec.sh](/Users/abasaran/Desktop/PhD/PaLaDiN/scripts/test_mvtec.sh:1).

Use `PALADIN_MVTEC_PATH` here too if your MVTec dataset is not under `./data/mvtec`.

### Evaluate on VisA

```bash
bash ./scripts/test_visa.sh paladin
```

For the top-k scoring variant:

```bash
bash ./scripts/test_visa.sh paladintopk
```

This script currently expects the checkpoint at:

```text
./code/visa_paladin/train_visa/epoch_10/pytorch_model.pt
```

If your checkpoint is elsewhere, edit [scripts/test_visa.sh](/Users/abasaran/Desktop/PhD/PaLaDiN/scripts/test_visa.sh:1).

Use `PALADIN_VISA_PATH` here too if your VisA dataset is not under `./data/visa`.

---

## Citation

If you use PALADIN in your research, please cite:

```bibtex
@InProceedings{Basaran_2026_CVPR,
  author    = {Basaran, Arda},
  title     = {PALADIN: Prompt-Aligned Localization and Anomaly Detection with DINOv3},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2026},
  pages     = {7693--7701}
}
```

---

## Acknowledgements

PALADIN builds on DINOv3, OpenCLIP, MVTec-AD, and VisA. We thank the authors and maintainers of these resources for making their models, datasets, and code available to the research community.
