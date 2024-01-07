import pandas as pd
import numpy as np
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# Load the kidney dataset
df = pd.read_csv('kidney.csv')
X = df.drop('CLASS', axis=1)
y = df['CLASS']

# Create an instance of the DecisionTreeClassifier class
dt = DecisionTreeClassifier(max_depth=None)

# Fit the decision tree
dt.fit(X.values, y.values)

# Define the class names
class_names = ['0', '1', '2']

# Generate the graphviz representation of the decision tree
dot_data = export_graphviz(dt, out_file=None, feature_names=X.columns, class_names=class_names, filled=True, rounded=True)

# Create a graph from the dot data
graph = graphviz.Source(dot_data, format='png')

# Save the graph to a file
graph_file = 'decision_tree.png'
graph.render(filename=graph_file, cleanup=True)

# Display the file path
print(f"Graph saved as {graph_file}")
