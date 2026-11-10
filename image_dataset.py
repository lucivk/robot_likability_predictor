import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from PIL import Image

class RobotDataset(Dataset):
    def __init__(self, image_folder, excel_file, transform):
        self.image_folder = image_folder
        self.df = pd.read_excel(excel_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Obtain image names and labels from Excel file
        image_name = self.df.iloc[idx]['picture_name']
        label = self.df.iloc[idx]['likability']
        label = torch.tensor(label, dtype=torch.float32)
        # Build the image path adding the ".jpg" extension
        image_path = os.path.join(self.image_folder, f'{image_name}.jpg')
        # Load image
        image = Image.open(image_path).convert('RGB')
        # Apply transforms if exist
        if self.transform:
            image = self.transform(image)
        return image, label

class TestImageDataset(Dataset):
    def __init__(self, image_folder, transform):
        self.image_paths = sorted(Path(image_folder).glob('*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = str(self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        image_name = Path(image_path).name
        if self.transform:
            image = self.transform(image)
        return image, image_name


def split_data(image_folder, excel_file, train_size, img_size):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = RobotDataset(image_folder, excel_file, train_transform)
    test_dataset = RobotDataset(image_folder, excel_file, test_transform)
    indices = torch.randperm(len(train_dataset)).tolist()
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(test_dataset, test_indices)
    return train_dataset, test_dataset

def save_split(base_dir, train_dataset, test_dataset):
    torch.save(train_dataset, f'{base_dir}/train_dataset.pt')
    torch.save(test_dataset, f'{base_dir}/test_dataset.pt')
    return

def load_split(base_dir):
    train_dataset = torch.load(f'{base_dir}/train_dataset.pt', weights_only=False)
    test_dataset = torch.load(f'{base_dir}/test_dataset.pt', weights_only=False)
    return train_dataset, test_dataset