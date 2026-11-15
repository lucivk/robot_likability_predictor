import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from image_dataset import TestImageDataset, split_data, save_split, load_split
from networks import SimpleNet, EfficientNet, DenseNet
from trainer import train_model, evaluate_model, test_model_no_labels
from export_results import plot_losses, export_excel, export_excel_test
import torch

def training(model, train_dataset, test_dataset, batch_size, num_epochs, learning_rate, augmentation, device):
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    test_loss, epochs, train_losses, test_losses = train_model(model, train_loader,
        test_loader, num_epochs, learning_rate, augmentation, device)
    return test_loss, epochs, train_losses, test_losses

def testing_no_labels(model, data_dir, batch_size, img_size, device):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = TestImageDataset(data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    predictions = test_model_no_labels(model, test_loader, device)
    return predictions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--split", choices=["new", "load", "none"], default="load")
    parser.add_argument("--model", choices=["SimpleNet", "EfficientNet", "DenseNet"])
    parser.add_argument("--aug", choices=["Y", "N"], default="N")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 224
    batch_size = 16
    learning_rate = 0.001 # 0.001 SimpleNet, 0.01 EfficientNet, 0.001 DenseNet
    num_epochs = 50

    if args.model == 'SimpleNet':
        model = SimpleNet(in_channels=3).to(device)
    elif args.model == 'EfficientNet':
        model = EfficientNet().to(device)
    elif args.model == 'DenseNet':
        model = DenseNet().to(device)
    else:
        raise Exception('Must insert a valid model')
    name = model.get_name()

    if args.mode == 'train':
        if args.split == 'new':
            train_dataset, test_dataset = split_data('Robot_images',
           'data_Uncanny_Valley.xlsx', 60, img_size, args.aug)
            save_split('Train_test_dataset', train_dataset, test_dataset, args.aug)
        elif args.split == 'load':
            train_dataset, test_dataset = load_split('Train_test_dataset', args.aug)
        else:
            raise Exception('Must specify a split in train mode')
        test_loss, epochs, train_losses, test_losses = training(model, train_dataset, test_dataset,
            batch_size, num_epochs, learning_rate, args.aug, device)
        plot_losses(name, epochs, train_losses, test_losses, learning_rate, args.aug, 'Results/Plots')
    else:
        model.load('Models', learning_rate, args.aug)
        if args.split == 'new':
            raise Exception('Cannot split in test mode')
        elif args.split == 'load':
            _, test_dataset = load_split('Train_test_dataset', args.aug)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
            test_loss, targets, predictions = evaluate_model(model, test_loader, device)
            print(f'Targets: {targets}, Predictions: {predictions}')
            export_excel(name, learning_rate, args.aug, test_loss, targets, predictions, 'Results/Tables')
        else:
            predictions = testing_no_labels(model, 'Test_images', batch_size, img_size, device)
            export_excel_test(name, predictions, 'Results/Tables')
    return

if __name__ == '__main__':
    main()