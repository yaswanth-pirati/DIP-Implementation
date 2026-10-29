# DIP-Implementation

## IACC: Underwater Image Enhancement

This project implements an underwater image enhancement approach designed to improve the visual quality of underwater images affected by colour distortion, haze, scattering, and illumination variations.

The implementation is developed using Python and PyTorch and includes separate modules for data handling, training, loss functions, model components, utilities, and testing.

## Dataset

This project uses the **UIEB (Underwater Image Enhancement Benchmark)** dataset.

### Dataset Source

Kaggle:  
https://www.kaggle.com/datasets/larjeck/uieb-dataset-reference

The complete dataset is **not included in this GitHub repository** because of its large size.

Please download the dataset from Kaggle and configure the dataset location according to the project configuration.

## Project Structure

````text
DIP-Implementation/
│
├── data/
│   ├── __init__.py
│   └── uieb.py
│
├── engine/
│   ├── __init__.py
│   ├── dehaze.py
│   ├── ema.py
│   ├── l1.py
│   ├── l2.py
│   └── multi_stage.py
│
├── log/
│
├── loss/
│   ├── __init__.py
│   └── perceptual.py
│
├── model/
│   ├── __init__.py
│   └── base.py
│
├── utils/
│   ├── __init__.py
│   ├── common_utils.py
│   └── scheduler_utils.py
│
├── config.yaml
├── requirements.txt
├── run.py
├── test.py
├── README.md
└── .gitignore