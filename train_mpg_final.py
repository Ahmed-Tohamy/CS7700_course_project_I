import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from mlp import MultilayerPerceptron, Layer, Tanh, Linear, SquaredError, batch_generator

# Set random seed for reproducibility
np.random.seed(42)

# Download the Vehicle MPG dataset directly
print("\nDownloading Vehicle MPG dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
local_file_path = "auto-mpg.data"

# Download the dataset with SSL verification disabled
response = requests.get(url, verify=False)
if response.status_code == 200:
    with open(local_file_path, 'wb') as f:
        f.write(response.content)
    print(f"Dataset downloaded successfully and saved to {local_file_path}")
else:
    raise Exception("Failed to download the dataset.")

# Load the Vehicle MPG dataset locally
print("\nLoading Vehicle MPG dataset from local file...")
column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
df = pd.read_csv(local_file_path, sep=r'\s+', names=column_names, na_values='?')

# Remove any missing values
df = df.dropna()

# Convert categorical variables
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")

# Extract features (X) and target (y)
X = df.drop(columns=["mpg", "car name"]).values
y = df["mpg"].values.reshape(-1, 1)

# Split dataset into training, validation, and test sets (70-15-15)
def manual_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """Splits dataset into training, validation, and test sets manually without sklearn."""
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    train_size = int(train_ratio * len(X))
    val_size = int(val_ratio * len(X))

    X_train, y_train = X[indices[:train_size]], y[indices[:train_size]]
    X_val, y_val = X[indices[train_size:train_size + val_size]], y[indices[train_size:train_size + val_size]]
    X_test, y_test = X[indices[train_size + val_size:]], y[indices[train_size + val_size:]]

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = manual_split(X, y)

print(f"\nData Split: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples.")

# Standardize features and target (zero mean, unit variance)
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
y_mean, y_std = y_train.mean(), y_train.std()

# Standardize X
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Standardize y
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

# Define MLP model for regression
mlp = MultilayerPerceptron([
    Layer(fan_in=X.shape[1], fan_out=128, activation_function=Tanh(), dropout_rate=0.2, weight_decay=1e-5),
    Layer(fan_in=128, fan_out=64, activation_function=Tanh(), dropout_rate=0.1, weight_decay=1e-5),
    Layer(fan_in=64, fan_out=32, activation_function=Tanh()),
    Layer(fan_in=32, fan_out=1, activation_function=Linear())  # Output layer for regression
])

loss_func = SquaredError()

# Train MLP
learning_rate = 0.01
epochs = 300
batch_size = 32
rmsprop = True  # Enable RMSProp optimizer

print(f"\nTraining MLP with {epochs} epochs, batch size {batch_size}, and RMSProp optimization...")

train_loss, val_loss = mlp.train(
    train_x=X_train, train_y=y_train,
    val_x=X_val, val_y=y_val,
    loss_func=loss_func,
    learning_rate=learning_rate,
    batch_size=batch_size,
    epochs=epochs,
    rmsprop=rmsprop
)

# Evaluate model on test set
y_pred_test = mlp.forward(X_test, training=False)
final_test_loss = loss_func.loss(y_test, y_pred_test)

print(f"\nFinal Test Loss (Mean Squared Error): {final_test_loss:.4f}")

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss for Vehicle MPG Regression")
plt.legend()
plt.grid(True)
plt.show()

# Predicted vs. actual MPG (10 samples)
sample_indices = np.random.choice(len(X_test), 10, replace=False)
predicted_vs_actual = pd.DataFrame({
    "True MPG": y_test[sample_indices].flatten() * y_std + y_mean,  # Convert back to original scale
    "Predicted MPG": y_pred_test[sample_indices].flatten() * y_std + y_mean
})

print("\nSample Predictions (True vs. Predicted MPG):")
print(predicted_vs_actual)

# Save results for report
predicted_vs_actual.to_csv("mpg_predictions_optimized.csv", index=False)