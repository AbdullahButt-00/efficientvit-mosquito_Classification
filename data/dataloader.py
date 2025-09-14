import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
# Import the visualization helper
from utils.visualization import show_batch

class MosquitoDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['filepath']
        label = self.label2idx[self.data.iloc[idx]['label']]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(train_csv, test_csv, batch_size=20):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = MosquitoDataset(train_csv, transform)
    test_dataset  = MosquitoDataset(test_csv, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.label2idx


def visualize_samples(loader, label2idx, title="Batch"):
    idx2label = {v: k for k, v in label2idx.items()}
    images, labels = next(iter(loader))
    show_batch(images, labels, idx2label, title=title)
