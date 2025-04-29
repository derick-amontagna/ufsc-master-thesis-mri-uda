# Init
This repository contains the codebase and resources for a master's thesis project focused on **domain adaptation unsupervised** in medical imaging, particularly involving MRI scans. It includes pipelines for data preprocessing, model training, evaluation, and domain shift experiments using tools like PyTorch, ANTsPy, and more.

## 🧠 Project Focus

This project investigates:
- **Domain shift problems** in medical imaging
- Transfer learning and domain adaptation using `pytorch-adapt`
- Skull stripping, bias correction, and spatial normalization for MRI
- Evaluation of model generalization across source and target domains

## 📦 Installation

> This project uses [Poetry](https://python-poetry.org/) for dependency management.
# 📁 Project Structure
## 🧪 Preprocess (`run_test_validador.ipynb`)

- Skull stripping using HD-BET or DeepBET
- Bias field correction via ANTsPy/SimpleITK
- Co-registration and spatial alignment

## Exp(`run python main.py --exp_name {runs/runs/ge_auc/ge_philips_atdoc_auc.json}`)
- Model training using PyTorch + Torchvision
- Domain adaptation with `pytorch` and `pytorch-adapt`

## Get Results (`run_test_validador`)
- Metric tracking with CometML

