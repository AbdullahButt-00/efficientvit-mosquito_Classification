# EfficientViT Mosquito Classification 

This project implements **EfficientViT** models for mosquito classification using **4 datasets**.  
It includes a modular pipeline for training, evaluation, and visualization.

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

## 🚀 Features
- Modular dataloaders for multiple datasets  
- Training & evaluation with EfficientViT  
- Batch visualization utilities  
- ONNX export support  
- Easy scaling to new datasets  

## 📊 Datasets
The pipeline is designed to handle **4 mosquito datasets**, with paths configured in `configs/dataset_paths.py`.

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/efficientvit_mosquito.git
cd efficientvit_mosquito
pip install -r requirements.txt
```

## ▶️ Usage
### Training
```bash
python main_train.py
```

### Evaluation
```bash
python main_eval.py
```

## 💡 Project Idea
Mosquito-borne diseases like **dengue, malaria, and Zika** are major global health issues.  
This project explores the use of **lightweight ViT models (EfficientViT)** for efficient mosquito species classification, enabling deployment in **resource-constrained environments** such as mobile devices.

## 📌 Requirements
See [`requirements.txt`](requirements.txt).

---

🔬 **Research Direction**: This repo can serve as a baseline for mosquito classification using **efficient vision transformers**.  
🤝 Contributions are welcome!
