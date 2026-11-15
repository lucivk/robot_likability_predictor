from torch import optim
from torch import nn
import torch

def train_model(model, train_loader, test_loader, num_epochs, learning_rate, augmentation, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    epochs = []
    patience = 5
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_samples = 0
        for images, labels in train_loader:
            # Move image and label to the device (GPU/CPU)
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            # Forward pass: compute the model output
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass: compute the gradients
            loss.backward()
            # Optimization step: update the model parameters
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            num_samples += labels.size(0)
        train_loss /= num_samples
        print(f"Epoch [{epoch + 1}], Train loss: {train_loss:.2f}")
        epochs.append(epoch + 1)
        train_losses.append(train_loss)
        # Evaluate model for early stopping
        test_loss, _, _ = evaluate_model(model, test_loader, device)
        test_losses.append(test_loss)
        # Save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            model.save('Models', learning_rate, augmentation)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        # Early stopping
        if epochs_without_improvement >= patience:
            break
    print(f"Training finished. Best loss: {best_test_loss:.2f}")
    return best_test_loss, epochs, train_losses, test_losses

def evaluate_model(model, test_loader, device):
    criterion = nn.MSELoss()
    model.eval()
    label_list = []
    prediction_list = []
    test_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Move image and label to the device (GPU/CPU)
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1) # Add an extra dimension to compute the loss
            outputs = model(images)
            label_list.extend(labels.squeeze(1).cpu().tolist())
            prediction_list.extend(outputs.squeeze(1).cpu().tolist())
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0) # Add losses without dividing by batch_size
            num_samples += labels.size(0)
    test_loss /= num_samples
    print(f"Test Loss: {test_loss:.2f}")
    return test_loss, label_list, prediction_list

def test_model_no_labels(model, test_loader, device):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for images, image_names in test_loader:
            images = images.to(device)
            predicts = model(images)
            for name, pred in zip(image_names, predicts):
                predictions[name] = pred.item()
    print(f"Predictions: {predictions}")
    return predictions