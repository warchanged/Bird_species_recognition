import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define the DNN model using PyTorch
class BirdDNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers=[256, 128], dropout_rate=0.3):
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
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

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

def load_model_and_predict(model_path, features, model_architecture, num_classes):
    """Load a trained model and make predictions."""
    # Create model with the same architecture
    input_dim = features.shape[1]
    model = BirdDNN(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=model_architecture['hidden_layers'],
        dropout_rate=model_architecture['dropout_rate']
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Convert features to tensor
    features_tensor = torch.FloatTensor(features)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.numpy()

def main():
    # Define the paths to the CSV files
    training_path = r"C:\Users\19395\Downloads\Project_6_Bird_species_recognition\birds-species-recognition\birds-species-recognition\training.csv"
    testing_path = r"C:\Users\19395\Downloads\Project_6_Bird_species_recognition\birds-species-recognition\birds-species-recognition\testing.csv"
    
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
    
    # Standardize features using the same scaler as during training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define the model architecture (should match the saved model)
    model_architecture = {
        'name': 'DNN-Medium',  # Change this to match your best model
        'hidden_layers': [512, 256, 128],
        'dropout_rate': 0.4
    }
    
    # Path to the saved model
    model_path = f"{model_architecture['name']}_best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    # Number of classes
    num_classes = len(np.unique(y_train))
    
    # Load model and make predictions
    print(f"Loading model from {model_path}...")
    predictions = load_model_and_predict(model_path, X_test_scaled, model_architecture, num_classes)
    
    # Adjust predictions to match original class IDs (add 1)
    predictions = predictions + 1
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Get class names
    class_id_to_name = {}
    for name, _, class_id, _ in training_data:
        if '\\' in name:
            class_name = name.split('\\')[0]
            if '.' in class_name:
                class_name = class_name.split('.', 1)[1]
            class_id_to_name[class_id] = class_name
    
    # Display some example predictions
    print("\nSample predictions:")
    for i in range(10):
        true_class_id = y_test[i]
        pred_class_id = predictions[i]
        true_class_name = class_id_to_name.get(true_class_id, "Unknown")
        pred_class_name = class_id_to_name.get(pred_class_id, "Unknown")
        
        print(f"Sample {i+1}:")
        print(f"  True: {true_class_name} (ID: {true_class_id})")
        print(f"  Predicted: {pred_class_name} (ID: {pred_class_id})")
        print(f"  {'✓' if true_class_id == pred_class_id else '✗'}")
        print()

if __name__ == "__main__":
    main()
