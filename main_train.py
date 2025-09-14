import torch
from configs.dataset_paths import dataset_paths, weight_base
from configs.training_config import training_config
from data.dataloader import get_dataloaders
from models.train import train_model

device = "cuda" if torch.cuda.is_available() else "cpu"

for dataset_name, paths in dataset_paths.items():
    print(f"\n📌 Training on {dataset_name}")
    train_loader, test_loader, label2idx = get_dataloaders(paths["train"], paths["test"], training_config["batch_size"])

    for model_name in training_config["models"]:
        print(f"\n Training {model_name} on {dataset_name}")
        model = train_model(model_name, train_loader,
                            training_config["num_classes"], device,
                            training_config["num_epochs"], training_config["learning_rate"])

        save_path = f"{weight_base}/{dataset_name}/" \
                    f"{dataset_name.lower()}_{model_name}.pth"
        torch.save(model.state_dict(), save_path)
        print(f" Saved {model_name} -> {save_path}")

