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

# Custom Dataset class for PyTorch
class BirdDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define a simple DNN model using PyTorch
class SimpleBirdDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleBirdDNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse the training and testing data
    print("Parsing training data...")
    training_data = parse_csv(training_path)
    print("Parsing testing data...")
    testing_data = parse_csv(testing_path)
    
    # Extract features and labels
    X_train = np.vstack([item[3] for item in training_data])
    y_train = np.array([item[2] - 1 for item in training_data])  # Subtract 1 because PyTorch expects 0-indexed classes
    X_test = np.vstack([item[3] for item in testing_data])
    y_test = np.array([item[2] - 1 for item in testing_data])
    
    # Preprocess the data
    print("Preprocessing data...")
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets and dataloaders
    train_dataset = BirdDataset(X_train_scaled, y_train)
    test_dataset = BirdDataset(X_test_scaled, y_test)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create the model
    input_dim = X_train_scaled.shape[1]
    num_classes = len(np.unique(y_train))
    model = SimpleBirdDNN(input_dim, num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model (just a few epochs for testing)
    print("Training model...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
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
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')
    
    # Evaluate the model
    print("Evaluating model...")
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
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # Get class names
    class_id_to_name = {}
    for name, _, class_id, _ in training_data:
        if '\\' in name:
            class_name = name.split('\\')[0]
            if '.' in class_name:
                class_name = class_name.split('.', 1)[1]
            class_id_to_name[class_id - 1] = class_name  # Subtract 1 for 0-indexed classes
    
    # Print classification report for a few classes
    unique_class_ids = sorted(class_id_to_name.keys())[:10]  # First 10 classes
    unique_class_names = [class_id_to_name[class_id] for class_id in unique_class_ids]
    
    print("\nClassification Report (First 10 classes):")
    report = classification_report(
        [label if label < 10 else -1 for label in all_labels],
        [pred if pred < 10 else -1 for pred in all_preds],
        labels=list(range(10)),
        target_names=unique_class_names,
        zero_division=0
    )
    print(report)
    
    print("Analysis complete.")
