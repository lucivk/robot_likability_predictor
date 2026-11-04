import os
import pandas as pd
import torch
import openpyxl
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import random_split
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
        image = Image.open(image_path)
        # Apply transforms if exist
        if self.transform:
            image = self.transform(image)
        return image, label

def split_data(train_size, img_size):
    transform = transforms.Compose([transforms.Resize((img_size, img_size)),transforms.ToTensor()])
    dataset = RobotDataset(image_folder="Robot_images", excel_file="data_Uncanny_Valley.xlsx", transform=transform)
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    return train_dataset, test_dataset

def save_dataset(train_dataset, test_dataset):
    torch.save(train_dataset, 'Train_test_data/train_dataset.pt')
    torch.save(test_dataset, 'Train_test_data/test_dataset.pt')

def load_dataset():
    train_dataset = torch.load('Train_test_data/train_dataset.pt', weights_only=False)
    test_dataset = torch.load('Train_test_data/test_dataset.pt', weights_only=False)
    return train_dataset, test_dataset