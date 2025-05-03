import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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

# Extract class information
train_classes = [item[1] for item in training_data]
test_classes = [item[1] for item in testing_data]

# Count occurrences of each class
train_class_counts = Counter(train_classes)
test_class_counts = Counter(test_classes)

# Print dataset statistics
print("\nDataset Statistics:")
print(f"Number of training samples: {len(training_data)}")
print(f"Number of testing samples: {len(testing_data)}")
print(f"Number of unique classes: {len(train_class_counts)}")
print(f"Feature vector dimension: {training_data[0][3].shape[0]}")

# Print some class examples
print("\nSample of class names:")
for i, (class_name, count) in enumerate(sorted(train_class_counts.items())[:10]):
    print(f"{i+1}. {class_name} - {count} training samples, {test_class_counts.get(class_name, 0)} testing samples")

# Check class distribution
print("\nClass distribution analysis:")
min_train_samples = min(train_class_counts.values())
max_train_samples = max(train_class_counts.values())
avg_train_samples = sum(train_class_counts.values()) / len(train_class_counts)

print(f"Minimum samples per class (training): {min_train_samples}")
print(f"Maximum samples per class (training): {max_train_samples}")
print(f"Average samples per class (training): {avg_train_samples:.2f}")

# Check feature statistics
all_train_features = np.vstack([item[3] for item in training_data])
feature_means = np.mean(all_train_features, axis=0)
feature_stds = np.std(all_train_features, axis=0)

print("\nFeature statistics:")
print(f"Feature mean range: [{np.min(feature_means):.4f}, {np.max(feature_means):.4f}]")
print(f"Feature std range: [{np.min(feature_stds):.4f}, {np.max(feature_stds):.4f}]")

# Plot class distribution (top 20 classes)
plt.figure(figsize=(12, 6))
top_classes = dict(sorted(train_class_counts.items(), key=lambda x: x[1], reverse=True)[:20])
plt.bar(range(len(top_classes)), list(top_classes.values()))
plt.xticks(range(len(top_classes)), list(top_classes.keys()), rotation=90)
plt.title('Distribution of Top 20 Classes (Training Set)')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.tight_layout()
plt.savefig('class_distribution.png')

print("\nAnalysis complete. Class distribution plot saved as 'class_distribution.png'")
