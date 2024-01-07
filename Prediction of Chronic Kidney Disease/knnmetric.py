import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset using pandas
data = pd.read_csv('kidney.csv')

# Split the data into features and target
features = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

# Split the dataset into training and test sets
split_ratio = 0.7
split_index = int(split_ratio * len(features))
train_features = features[:split_index]
train_target = target[:split_index]
test_features = features[split_index:]
test_target = target[split_index:]

# Define the number of nearest neighbors to consider
k = 3

# Define a function to calculate the Euclidean distance between two data points
def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i])**2
    return distance**0.5

# Define a function to find the k nearest neighbors of a given data point
def get_neighbors(train_data, train_labels, test_instance, k):
    distances = []
    for i in range(len(train_data)):
        dist = euclidean_distance(test_instance, train_data[i])
        distances.append((train_labels[i], dist))
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

# Define a function to make predictions for a given test dataset
def predict(train_data, train_labels, test_data, k):
    predictions = []
    for i in range(len(test_data)):
        neighbors = get_neighbors(train_data, train_labels, test_data[i], k)
        counts = {}
        for j in range(len(neighbors)):
            response = neighbors[j]
            if response in counts:
                counts[response] += 1
            else:
                counts[response] = 1
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        predictions.append(sorted_counts[0][0])
    return predictions

# Make predictions for the test dataset
predictions = predict(train_features, train_target, test_features, k)

# Calculate precision, recall, and F1 score using scikit-learn
precision = precision_score(test_target, predictions, average='weighted')
recall = recall_score(test_target, predictions, average='weighted')
f1 = f1_score(test_target, predictions, average='weighted')

# Print the results
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Calculate the confusion matrix
cm = confusion_matrix(test_target, predictions)

# Plot the confusion matrix
labels = ['0', '1', '2']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
