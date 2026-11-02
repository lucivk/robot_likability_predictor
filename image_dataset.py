import os
import pandas as pd
import torch
from torch.utils.data import Dataset
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