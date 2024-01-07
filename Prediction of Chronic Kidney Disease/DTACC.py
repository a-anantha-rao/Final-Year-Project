import pandas as pd
import numpy as np

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        
    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.tree_ = self._grow_tree(X, y)
        
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]
    
    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        
        for idx in range(X.shape[1]):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))
                gini = (i * gini_left + (m - i) * gini_right) / m
                
                if thresholds[i] == thresholds[i - 1]:
                    continue
                    
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
                    
        return best_idx, best_thr
    
    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = DecisionNode(value=predicted_class)
        
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] <= thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node = DecisionNode(feature=idx, threshold=thr, left=self._grow_tree(X_left, y_left, depth + 1),
                                    right=self._grow_tree(X_right, y_right, depth + 1))
                
        return node
    
    def _predict(self, inputs):
        node = self.tree_
        while not node.is_leaf_node():
            if inputs[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


df = pd.read_csv("kidney.csv")
X = df.drop(columns=["CLASS"]).values
y = df["CLASS"].values

train_size = int(0.7 * len(df))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create and train the decision tree model
dt = DecisionTree(max_depth=5)
dt.fit(X_train, y_train)

# Make prediction on the test set
y_pred = dt.predict(X_test)
d_accuracy = sum(y_pred == y_test) / len(y_test) *100
#print(d_accuracy)





