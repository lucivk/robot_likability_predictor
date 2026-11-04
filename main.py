import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from image_dataset import split_data
from networks import SimpleCNN
from trainer import train_model
import torch

def training(model, device):
    train_dataset, test_dataset = split_data(60, 224)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    learning_rate = 0.001
    num_epochs = 50
    test_loss = train_model(model, train_loader, test_loader, num_epochs, learning_rate, device)
    return test_loss

def testing(model, device, data_dir):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_dataset = ImageFolder(root=str(data_dir), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    predictions = []
    for image in test_loader:
        image = image.to(device)
        pred = model(image)
        predictions.append(pred)
    return predictions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model", type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == 'train':
        if args.model == 'SimpleCNN':
            model = SimpleCNN(in_channels=3).to(device)
        else:
            raise Exception('Invalid model')
        training(model, device)
    if args.mode == 'test' and args.data_dir is None:
        raise Exception('--data_dir is required when mode is test')
    else:
        if args.model == 'SimpleCNN':
            model = SimpleCNN(in_channels=3).to(device)
            model.load('Models/SimpleCNN.pth')
        else:
            raise Exception('Invalid model')
        predictions = testing(model, device, args.data_dir)
        print(predictions)
    return

if __name__ == '__main__':
    main()