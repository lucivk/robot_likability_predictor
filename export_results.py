import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import numpy as np
import pandas as pd

def plot_losses(model_name, epochs, train_losses, test_losses, learning_rate, augmentation, base_dir):
    plt.plot(epochs, train_losses, label='Train', color='blue', marker='o', linestyle='--')
    plt.plot(epochs, test_losses, label='Test', color='orange', marker='o')
    plt.xlabel("Epoch", fontsize='x-large')
    plt.ylabel("Loss", fontsize='x-large')
    plt.legend(fontsize='x-large', loc='upper right')
    plt.savefig(f'{base_dir}/{model_name}_lr{learning_rate}_aug{augmentation}_losses.png')
    plt.show()
    return

def calculate_metrics(model_name, targets, predictions):
    targets = np.array(targets)
    predictions = np.array(predictions)
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r_mse = root_mean_squared_error(targets, predictions)
    print(f'Model: {model_name}, R^2: {r2:.2f}, MAE: {mae:.2f}, RMSE: {r_mse:.2f}')
    return r2, mae, r_mse

def export_excel(model_name, learning_rate, augmentation, test_loss, targets, predictions, base_dir):
    r2, mae, r_mse = calculate_metrics(model_name, targets, predictions)
    df = pd.DataFrame({'Model': [model_name],
                       'Optimizer': 'Adam',
                       'Augmentation': [augmentation],
                       'Learning rate': [learning_rate],
                       'Test loss': [test_loss],
                       'R^2': [r2],
                       'MAE': [mae],
                       'RMSE': [r_mse]})
    df.to_excel(f'{base_dir}/{model_name}_lr{learning_rate}_aug{augmentation}_results.xlsx', index=False)
    return

def export_excel_test(model_name, predictions, base_dir):
    df = pd.DataFrame(list(predictions.items()), columns=['Image', 'Prediction'])
    df.to_excel(f'{base_dir}/{model_name}_test_no_labels.xlsx', index=False)
    return