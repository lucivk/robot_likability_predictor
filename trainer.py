from torch import optim
from torch import nn
import torch
import networks

def train_model(model, train_loader, test_loader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_test_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total_samples = 0
        for image, label in train_loader:
            # Move image and label to the device (GPU/CPU)
            image = image.to(device)
            label = label.to(device)
            label = label.unsqueeze(1)
            # Forward pass: compute the model output
            pred = model(image)
            loss = criterion(pred, label)
            # Backward pass: compute the gradients
            optimizer.zero_grad()
            loss.backward()
            # Optimization step: update the model parameters
            optimizer.step()
            train_loss += loss.item() * image.size(0)
            total_samples += image.size(0)
        train_loss /= total_samples
        print(f"Epoch [{epoch + 1}], Loss: {train_loss:.2f}")
        # Evaluate model for early stopping
        test_loss, _, _ = evaluate_model(model, test_loader, device)
        # Save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            model.save('Models')
        else:
            epochs_without_improvement += 1
        # Early stopping
        if epochs_without_improvement >= patience:
            break
    print(f"Training finished. Best loss: {best_test_loss:.2f}")
    return best_test_loss

def evaluate_model(model, test_loader, device):
    criterion = nn.MSELoss()
    model.eval()
    labels = []
    predicts = []
    test_loss = 0
    total_samples = 0
    with torch.no_grad():
        for image, label in test_loader:
            # Move image and label to the device (GPU/CPU)
            image = image.to(device)
            label = label.to(device)
            label = label.unsqueeze(1)
            pred = model(image)
            labels.append(label)
            predicts.append(pred)
            loss = criterion(pred, label)
            test_loss += loss.item() * image.size(0)
            total_samples += image.size(0)
    test_loss /= total_samples
    print(f"Test Loss: {test_loss:.2f}")
    return test_loss, labels, predicts