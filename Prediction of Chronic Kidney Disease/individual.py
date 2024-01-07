import matplotlib.pyplot as plt
import numpy as np

# Data for first algorithm
algorithm1 = 'KNN'
precision1 = 0.8
recall1 = 0.8
f1_score1 = 0.8

# Data for second algorithm
algorithm2 = 'Decision Tree'
precision2 = 0.84
recall2 = 0.85
f1_score2 = 0.84

# Plotting the first graph for Algorithm 1
plt.figure(figsize=(6, 4))
plt.bar([0], [precision1], color='#66C2A5', label='Precision')
plt.bar([1], [recall1], color='#FC8D62', label='Recall')
plt.bar([2], [f1_score1], color='#8DA0CB', label='F1-score')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title(f'Algorithm Performance - {algorithm1}')
plt.xticks([0, 1, 2], ['Precision', 'Recall', 'F1-score'])
plt.legend()
plt.tight_layout()
plt.savefig(f'static/performance_graph_{algorithm1}.png')
plt.close()

# Plotting the second graph for Algorithm 2
plt.figure(figsize=(6, 4))
plt.bar([0], [precision2], color='#66C2A5', label='Precision')
plt.bar([1], [recall2], color='#FC8D62', label='Recall')
plt.bar([2], [f1_score2], color='#8DA0CB', label='F1-score')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title(f'Algorithm Performance - {algorithm2}')
plt.xticks([0, 1, 2], ['Precision', 'Recall', 'F1-score'])
plt.legend()
plt.tight_layout()
plt.savefig(f'static/performance_graph_{algorithm2}.png')
plt.close()
