# EfficientViT Mosquito Classification 

This project applies **EfficientViT** to the task of mosquito species classification using **4 datasets** different datasets. The pipeline is modular, enabling training and evaluation of multiple EfficientViT variants (B0–B3, L1–L3) with standardised dataloaders, training loops, and evaluation scripts.
The goal is to benchmark EfficientViT models on real-world biological image data, measuring not just accuracy, precision, recall, and F1 score, but also inference speed — a key factor for deploying lightweight vision transformers in resource-constrained environments.

## 📂 Project Structure
```
efficientvit_mosquito/
│
├── configs/
│   ├── dataset_paths.py      # Paths for all 4 datasets
│   ├── training_config.py    # Training hyperparameters
│
├── data/
│   ├── dataloader.py         # Custom Dataset + Dataloaders
│
├── models/
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Evaluation metrics & inference time
│
├── utils/
│   ├── visualization.py      # Batch visualization, etc.
│
├── main_train.py             # Train all datasets & models
├── main_eval.py              # Evaluate all trained models
├── requirements.txt          # Dependencies
```

## Features
- Modular dataloaders for multiple datasets  
- Training & evaluation with EfficientViT  
- Batch visualisation utilities  
- Easy scaling to new datasets  

## Datasets
The pipeline is designed to handle **4 mosquito datasets**, with paths configured in `configs/dataset_paths.py`.

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/efficientvit_mosquito.git
cd efficientvit_mosquito
pip install -r requirements.txt
```

## Usage
### Training
```bash
python main_train.py
```

### Evaluation
```bash
python main_eval.py
```

## 📌 Requirements
See [`requirements.txt`](requirements.txt).

---
## 📊 Dataset Details

| Dataset   | Classes | Train (Images per Class)                          | Test (Images per Class)                           | Training Images | Test Images | Total |
|-----------|---------|--------------------------------------------------|--------------------------------------------------|----------------|-------------|-------|
| Dataset_1 | 3       | aedes: 150, anopheles: 104, culex: 99            | aedes: 38, anopheles: 26, culex: 25              | 353            | 89          | 442   |
| Dataset_2 | 3       | culex: 960, aedes: 720, anopheles: 432           | culex: 240, aedes: 180, anopheles: 108           | 2112           | 528         | 2640  |
| Dataset_3 | 3       | anopheles: 400, culex: 400, aedes: 400           | anopheles: 400, culex: 400, aedes: 400           | 1200           | 1200        | 2400  |
| Dataset_4 | 3       | culex: 1459, aedes: 1270, anopheles: 936         | culex: 665, aedes: 618, anopheles: 534           | 3665           | 1817        | 5482  |




🔬 **Research Direction**: This repo can serve as a baseline for mosquito classification using **efficient vision transformers**.  


