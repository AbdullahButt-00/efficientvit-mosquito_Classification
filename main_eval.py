import torch
from configs.dataset_paths import dataset_paths, weight_base
from configs.training_config import training_config
from data.dataloader import get_dataloaders
from models.evaluate import evaluate_model
from efficientvit.cls_model_zoo import create_efficientvit_cls_model

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = training_config["num_classes"]

for dataset_name, paths in dataset_paths.items():
    print(f"\n Evaluating on {dataset_name}")
    _, test_loader, _ = get_dataloaders(paths["train"], paths["test"], training_config["batch_size"])

    for model_name in training_config["models"]:
        weight_path = f"{weight_base}/{dataset_name}/{dataset_name.lower()}_{model_name}.pth"
        model = create_efficientvit_cls_model(model_name, pretrained=False, num_classes=num_classes)
        state_dict = torch.load(weight_path, map_location=device)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        metrics = evaluate_model(model, test_loader, num_classes, device)
        print(f"{model_name} -> {metrics}")

