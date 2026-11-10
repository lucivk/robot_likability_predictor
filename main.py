import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from image_dataset import TestImageDataset, split_data, save_split, load_split
from networks import SimpleCNN, DenseNet
from trainer import train_model, evaluate_model, test_model_no_labels
import torch

def training(model, train_dataset, test_dataset, device):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    learning_rate = 0.0001
    num_epochs = 50
    test_loss, epochs, train_losses, test_losses = train_model(model, train_loader, test_loader, num_epochs, learning_rate, device)
    return test_loss, epochs, train_losses, test_losses

def testing_no_labels(model, data_dir, device):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_dataset = TestImageDataset(data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    predictions = test_model_no_labels(model, test_loader, device)
    return predictions

def plot_losses(model_name, epochs, train_losses, test_losses):
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, test_losses, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f'{model_name} train and test losses')
    plt.savefig(f'Plots/{model_name}_losses.png')
    plt.show()
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--split", choices=["new", "load", "none"], default="load")
    parser.add_argument("--model", choices=["SimpleCNN", "DenseNet"])
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == 'SimpleCNN':
        model = SimpleCNN(in_channels=3).to(device)
    elif args.model == 'DenseNet':
        model = DenseNet().to(device)
    else:
        raise Exception('Must insert a valid model')

    if args.mode == 'train':
        if args.split == 'new':
            train_dataset, test_dataset = split_data('Robot_images', 'data_Uncanny_Valley.xlsx', 60, 224)
            save_split('Train_test_dataset', train_dataset, test_dataset)
        elif args.split == 'load':
            train_dataset, test_dataset = load_split('Train_test_dataset')
        else:
            raise Exception('Must specify a split in train mode')
        _, epochs, train_losses, test_losses = training(model, train_dataset, test_dataset, device)
        name = model.get_name()
        plot_losses(name, epochs, train_losses, test_losses)
    else:
        model.load('Models')
        if args.split == 'new':
            raise Exception('Cannot split in test mode')
        elif args.split == 'load':
            _, test_dataset = load_split('Train_test_dataset')
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            _, predictions = evaluate_model(model, test_loader, device)
            print(f"Predictions: {predictions}")
        else:
            testing_no_labels(model, 'Test_images', device)
    return

if __name__ == '__main__':
    main()