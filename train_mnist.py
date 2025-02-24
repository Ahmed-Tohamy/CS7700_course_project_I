import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import kaggle
from mlp import MultilayerPerceptron, Layer, Relu, Softmax, CrossEntropy

# Step 1: Download and Prepare MNIST Data
mnist_data_dir = "mnist_data/"
os.makedirs(mnist_data_dir, exist_ok=True)

KAGGLE_DATASET = "avnishnish/mnist-original"
MNIST_MAT_FILE = os.path.join(mnist_data_dir, "mnist-original.mat")

def download_mnist_kaggle():
    """Downloads the MNIST dataset from Kaggle if not already present."""
    if os.path.exists(MNIST_MAT_FILE):
        print(f"{MNIST_MAT_FILE} already exists. Skipping download.")
        return
    print(f"\nDownloading MNIST dataset from Kaggle...")
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=mnist_data_dir, unzip=True)
    if not os.path.exists(MNIST_MAT_FILE):
        raise FileNotFoundError("ERROR: MNIST dataset download failed!")
    print(f"MNIST dataset successfully downloaded: {MNIST_MAT_FILE}")

download_mnist_kaggle()

# Step 2: Convert MNIST Data to NumPy Format
def convert_mat_to_mnist():
    """Converts mnist-original.mat to NumPy format."""
    try:
        print(f"\nConverting {MNIST_MAT_FILE} to MNIST format...")
        mat = scipy.io.loadmat(MNIST_MAT_FILE)
        images = mat.get("data").T.reshape(-1, 28 * 28).astype(np.float32)
        labels = mat.get("label")[0].astype(int)

        # Train-test split
        train_images, test_images = images[:60000], images[60000:]
        train_labels, test_labels = labels[:60000], labels[60000:]

        np.save(os.path.join(mnist_data_dir, "train_images.npy"), train_images)
        np.save(os.path.join(mnist_data_dir, "train_labels.npy"), train_labels)
        np.save(os.path.join(mnist_data_dir, "test_images.npy"), test_images)
        np.save(os.path.join(mnist_data_dir, "test_labels.npy"), test_labels)

        print("MNIST dataset successfully converted and saved.")
    except Exception as e:
        print(f"ERROR: Failed to process dataset: {e}")
        exit(1)

convert_mat_to_mnist()

# Step 3: Load and Standardize Data
def load_mnist():
    """Loads MNIST dataset from saved NumPy files."""
    try:
        train_images = np.load(os.path.join(mnist_data_dir, "train_images.npy"))
        train_labels = np.load(os.path.join(mnist_data_dir, "train_labels.npy"))
        test_images = np.load(os.path.join(mnist_data_dir, "test_images.npy"))
        test_labels = np.load(os.path.join(mnist_data_dir, "test_labels.npy"))
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        exit(1)
    return train_images, train_labels, test_images, test_labels

train_x, train_y, test_x, test_y = load_mnist()

def standardize_data(train_x, test_x):
    """Standardizes data (zero mean, unit variance)."""
    mean, std = np.mean(train_x, axis=0), np.std(train_x, axis=0)
    std[std == 0] = 1  # Prevent division by zero
    return (train_x - mean) / std, (test_x - mean) / std

train_x, test_x = standardize_data(train_x, test_x)

def one_hot_encode(labels, num_classes=10):
    """Converts integer labels to one-hot encoding."""
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

train_y, test_y = one_hot_encode(train_y), one_hot_encode(test_y)

# Split into training and validation sets
train_x, val_x = train_x[:55000], train_x[55000:]
train_y, val_y = train_y[:55000], train_y[55000:]

print(f"Data Loaded: {train_x.shape[0]} train, {val_x.shape[0]} validation, {test_x.shape[0]} test samples.")

# Step 4: Define and Train MLP Model
layers = [
    Layer(784, 128, Relu(), dropout_rate=0.2),
    Layer(128, 64, Relu(), dropout_rate=0.15),
    Layer(64, 10, Softmax())
]

mlp = MultilayerPerceptron(layers)

def categorical_accuracy(y_true, y_pred):
    """Computes classification accuracy."""
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

epochs = 160
batch_size = 256
learning_rate = 0.0005  # Adjusted for better stability
loss_function = CrossEntropy()

# Implement Learning Rate Decay
decay_factor = 0.95  # Reduce learning rate every 10 epochs
for epoch in range(epochs):
    if epoch % 10 == 0 and epoch > 0:
        learning_rate *= decay_factor

print(f"Training MLP with {epochs} epochs, batch size {batch_size}, learning rate {learning_rate}")

training_loss, validation_loss = mlp.train(
    train_x, train_y, val_x, val_y, loss_function,
    learning_rate, batch_size, epochs, rmsprop=True
)

# Step 5: Evaluate Model and Visualize Results
test_output = mlp.forward(test_x, training=False)
accuracy = categorical_accuracy(test_y, test_output)
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

# Plot training and validation loss curves
plt.figure(figsize=(8, 5))
plt.plot(training_loss, label="Training Loss")
plt.plot(validation_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss for MNIST")
plt.legend()
plt.grid(True)
plt.show()

# Visualize Sample Predictions
selected_indices = []
for digit in range(10):
    indices = np.where(np.argmax(test_y, axis=1) == digit)[0]
    selected_indices.append(indices[0])  # Take the first occurrence of each digit

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    img = test_x[selected_indices[i]].reshape(28, 28)
    true_label = np.argmax(test_y[selected_indices[i]])
    pred_label = np.argmax(test_output[selected_indices[i]])
    ax.imshow(img, cmap="gray")
    ax.set_title(f"True: {true_label}\nPred: {pred_label}")
    ax.axis("off")
plt.show()

# Plot True vs. Predicted Label Distribution
true_labels, predicted_labels = np.argmax(test_y, axis=1), np.argmax(test_output, axis=1)
plt.figure(figsize=(8, 5))
plt.hist([true_labels, predicted_labels], bins=np.arange(11) - 0.5, label=["True Labels", "Predicted Labels"], alpha=0.7)
plt.xticks(range(10))
plt.xlabel("Digit")
plt.ylabel("Count")
plt.legend()
plt.title("True vs. Predicted Label Distribution")
plt.grid(True)
plt.show()