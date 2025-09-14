import torch, time
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

def evaluate_model(model, test_loader, num_classes, device):
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
    recall_metric    = MulticlassRecall(num_classes=num_classes, average="macro").to(device)
    f1_metric        = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)

    model.eval()
    correct, total = 0, 0
    all_preds, all_labels, inference_times = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if device == "cuda": torch.cuda.synchronize()
            start = time.time()
            outputs = model(images)
            if device == "cuda": torch.cuda.synchronize()
            end = time.time()

            preds = outputs.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.append(preds)
            all_labels.append(labels)
            inference_times.append((end - start) / images.size(0) * 1000.0)

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    precision = precision_metric(all_preds, all_labels).item()
    recall    = recall_metric(all_preds, all_labels).item()
    f1        = f1_metric(all_preds, all_labels).item()
    accuracy  = 100.0 * correct / total
    avg_infer = sum(inference_times) / len(inference_times)

    return {"Accuracy (%)": round(accuracy,2), "Precision": round(precision,4),
            "Recall": round(recall,4), "F1 Score": round(f1,4),
            "Inference (ms)": round(avg_infer,4)}

