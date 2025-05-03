import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import warnings
warnings.filterwarnings('ignore')

# Define the paths to the CSV files
training_path = r"C:\Users\19395\Downloads\Project_6_Bird_species_recognition\birds-species-recognition\birds-species-recognition\training.csv"
testing_path = r"C:\Users\19395\Downloads\Project_6_Bird_species_recognition\birds-species-recognition\birds-species-recognition\testing.csv"

def parse_csv(file_path):
    """Parse the CSV file and extract image paths, class names, class IDs, and feature vectors."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',', 2)  # Split at the first two commas
            if len(parts) >= 3:
                image_path = parts[0]
                class_id = int(parts[1])

                # Extract class name from the image_path
                if '\\' in image_path:
                    class_name = image_path.split('\\')[0]
                    # Remove the numeric prefix if present (e.g., "001.Black_footed_Albatross" -> "Black_footed_Albatross")
                    if '.' in class_name:
                        class_name = class_name.split('.', 1)[1]
                else:
                    class_name = "Unknown"

                # Parse the feature vector
                features = np.array([float(x) for x in parts[2].split(',')])

                data.append((image_path, class_name, class_id, features))
    return data

# Parse the training and testing data
print("Parsing training data...")
training_data = parse_csv(training_path)
print("Parsing testing data...")
testing_data = parse_csv(testing_path)

# Extract features and labels
X_train = np.vstack([item[3] for item in training_data])
y_train = np.array([item[2] for item in training_data])
X_test = np.vstack([item[3] for item in testing_data])
y_test = np.array([item[2] for item in testing_data])

# Custom Dataset class for PyTorch
class BirdDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define DNN model using PyTorch
class BirdDNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers=[1024, 512, 256], dropout_rate=0.5):
        super(BirdDNN, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
            layers.append(nn.Dropout(dropout_rate))

        # Output layer - 200个类别
        layers.append(nn.Linear(hidden_layers[-1], num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=15):
    model.to(device)

    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)

        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | '
              f'Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}')

        # Early stopping
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                model.load_state_dict(best_model_state)
                break

    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# Function to evaluate the model
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)

    return test_loss, accuracy, all_preds, all_labels

# Preprocess the data
print("Preprocessing data...")
# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Adjust labels to be 0-indexed for PyTorch
y_train_adjusted = y_train - 1  # Subtract 1 because class IDs start from 1
y_test_adjusted = y_test - 1

# Split training data into train and validation sets
val_size = 0.1
val_indices = np.random.choice(len(X_train_scaled), int(val_size * len(X_train_scaled)), replace=False)
train_indices = np.array([i for i in range(len(X_train_scaled)) if i not in val_indices])

X_val = X_train_scaled[val_indices]
y_val = y_train_adjusted[val_indices]
X_train_final = X_train_scaled[train_indices]
y_train_final = y_train_adjusted[train_indices]

print(f"Training data shape: {X_train_final.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Testing data shape: {X_test_scaled.shape}")

# Create datasets and dataloaders
train_dataset = BirdDataset(X_train_final, y_train_final)
val_dataset = BirdDataset(X_val, y_val)
test_dataset = BirdDataset(X_test_scaled, y_test_adjusted)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create and train the DNN model
print("Training DNN model...")
input_dim = X_train_scaled.shape[1]
num_classes = len(np.unique(y_train))

# Define model architectures to try
model_architectures = [
    {'name': 'DNN-Small', 'hidden_layers': [512, 256], 'dropout_rate': 0.3},
    {'name': 'DNN-Medium', 'hidden_layers': [1024, 512, 256], 'dropout_rate': 0.4},
    {'name': 'DNN-Large', 'hidden_layers': [2048, 1024, 512, 256], 'dropout_rate': 0.5}
]

results = []

for arch in model_architectures:
    print(f"\nTraining {arch['name']}...")
    model = BirdDNN(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=arch['hidden_layers'],
        dropout_rate=arch['dropout_rate']
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Train the model
    start_time = time.time()
    trained_model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=15
    )
    training_time = time.time() - start_time

    # Evaluate the model
    test_loss, test_acc, y_pred, y_true = evaluate_model(trained_model, test_loader, criterion, device)

    # Store results
    results.append({
        'name': arch['name'],
        'accuracy': test_acc,
        'test_loss': test_loss,
        'training_time': training_time,
        'model': trained_model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'y_pred': y_pred,
        'y_true': y_true
    })

    print(f"{arch['name']} - Test Accuracy: {test_acc:.4f}, Training Time: {training_time:.2f}s")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.title(f'{arch["name"]} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])

    plt.subplot(1, 2, 2)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title(f'{arch["name"]} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])

    plt.tight_layout()
    plt.savefig(f'{arch["name"]}_training_history.png')

# Find the best model
best_model_result = max(results, key=lambda x: x['accuracy'])
print(f"\nBest model: {best_model_result['name']} with accuracy: {best_model_result['accuracy']:.4f}")

# Generate detailed classification report for the best model
print("\nDetailed Classification Report for Best Model:")

# Get class names
class_id_to_name = {}
for name, _, class_id, _ in training_data:
    if '\\' in name:
        class_name = name.split('\\')[0]
        if '.' in class_name:
            class_name = class_name.split('.', 1)[1]
        class_id_to_name[class_id - 1] = class_name  # Subtract 1 for 0-indexed classes

# Get unique class names in order of class IDs
unique_class_ids = sorted(class_id_to_name.keys())
unique_class_names = [class_id_to_name[class_id] for class_id in unique_class_ids]

# Print classification report
y_true = best_model_result['y_true']
y_pred = best_model_result['y_pred']

# Calculate overall metrics
report = classification_report(y_true, y_pred, output_dict=True)
print(f"Precision (avg): {report['macro avg']['precision']:.4f}")
print(f"Recall (avg): {report['macro avg']['recall']:.4f}")
print(f"F1-score (avg): {report['macro avg']['f1-score']:.4f}")

# Print report for first 10 classes only
print("\nClassification Report (First 10 classes):")
# Filter predictions for first 10 classes
mask_true = np.array([label < 10 for label in y_true])
mask_pred = np.array([pred < 10 for pred in y_pred])
mask = mask_true & mask_pred

if np.sum(mask) > 0:
    y_true_subset = np.array(y_true)[mask]
    y_pred_subset = np.array(y_pred)[mask]
    class_report = classification_report(
        y_true_subset,
        y_pred_subset,
        target_names=unique_class_names[:10],
        zero_division=0
    )
    print(class_report)

# Plot confusion matrix (for a subset of classes for visibility)
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm[:10, :10], interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (First 10 Classes)')
plt.colorbar()
plt.xticks(np.arange(10), unique_class_names[:10], rotation=90)
plt.yticks(np.arange(10), unique_class_names[:10])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

# Compare model performances
plt.figure(figsize=(10, 6))
names = [r['name'] for r in results]
accuracies = [r['accuracy'] for r in results]
training_times = [r['training_time'] / 60 for r in results]  # Convert to minutes

plt.subplot(1, 2, 1)
plt.bar(names, accuracies)
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1.0)  # Adjust as needed

plt.subplot(1, 2, 2)
plt.bar(names, training_times)
plt.title('Training Time Comparison')
plt.xlabel('Model')
plt.ylabel('Training Time (minutes)')

plt.tight_layout()
plt.savefig('model_comparison.png')

# Save the best model
best_model_path = f"{best_model_result['name']}_best_model.pth"
torch.save(best_model_result['model'].state_dict(), best_model_path)
print(f"\nBest model saved to {best_model_path}")

print("\nAnalysis complete. Results saved as PNG files.")
