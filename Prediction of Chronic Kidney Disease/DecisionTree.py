import pandas as pd
import math
import random

class Node:
    def __init__(self, feature_index=None, threshold=None, label=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.label = label
        self.left_child = None
        self.right_child = None

def load_dataset(filename):
    df = pd.read_csv(filename)
    features = df[['AGE', 'WEIGHT', 'SG', 'Alb', 'eGFR', 'Wbc']].values
    target = df['CLASS'].values
    dataset = []
    for i in range(len(features)):
        dataset.append(list(features[i]) + [target[i]])
    return dataset

def gini_index(groups, classes):
    # Calculate the Gini index for a split dataset
    total_instances = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / total_instances)
    return gini

def split_dataset(dataset, feature_index, threshold):
    left, right = [], []
    for row in dataset:
        if row[feature_index] < threshold:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_feature_index, best_threshold, best_gini, best_groups = math.inf, math.inf, math.inf, None
    for feature_index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = split_dataset(dataset, feature_index, row[feature_index])
            gini = gini_index(groups, class_values)
            if gini < best_gini:
                best_feature_index, best_threshold, best_gini, best_groups = feature_index, row[feature_index], gini, groups
    return {'feature_index': best_feature_index, 'threshold': best_threshold, 'groups': best_groups}

def create_leaf_node(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def create_decision_tree(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = create_leaf_node(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = create_leaf_node(left), create_leaf_node(right)
        return
    if len(left) <= min_size:
        node['left'] = create_leaf_node(left)
    else:
        node['left'] = get_best_split(left)
        create_decision_tree(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = create_leaf_node(right)
    else:
        node['right'] = get_best_split(right)
        create_decision_tree(node['right'], max_depth, min_size, depth + 1)

def build_decision_tree(dataset, max_depth, min_size):
    root = get_best_split(dataset)
    create_decision_tree(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    if row[node['feature_index']] < node['threshold']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def calculate_accuracy(actual, predicted):
    correct = sum(1 for i in range(len(actual)) if actual[i] == predicted[i])
    return (correct / float(len(actual))) * 100.0

# Load the dataset
dataset = load_dataset('kidney.csv')

# Split the dataset into train and test sets
train_size = int(0.7 * len(dataset))
train_data = random.sample(dataset, train_size)
test_data = [data for data in dataset if data not in train_data]

# Build the decision tree
max_depth = 3
min_size = 10
decision_tree = build_decision_tree(train_data, max_depth, min_size)

# Make predictions on the test set
actual = [row[-1] for row in test_data]
predicted = [predict(decision_tree, row) for row in test_data]

# Calculate accuracy
accuracy = calculate_accuracy(actual, predicted)
#print(f"Accuracy: {accuracy}%")

# Option to enter patient details and predict
patient_data = [57,89, 1.02, 4, 80, 7800]  # Example patient data, update with actual values

predicted_class = predict(decision_tree, patient_data)
#print(f"Predicted Class for Patient: {predicted_class}")
