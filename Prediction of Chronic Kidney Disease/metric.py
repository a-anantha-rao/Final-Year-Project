import matplotlib.pyplot as plt
import numpy as np

#Precision, Recall, and F1-score data
algorithms = ['KNN', 'Decision Tree']
precision = [0.8, 0.84]
recall = [0.8, 0.85]
f1_score = [0.8, 0.84]

#Set the width of the bars
bar_width = 0.25

#Set the x position of the bars
r1 = np.arange(len(algorithms))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

#Plotting the graph
plt.bar(r1, precision, color='#66C2A5', width=bar_width, label='Precision')
plt.bar(r2, recall, color='#FC8D62', width=bar_width, label='Recall')
plt.bar(r3, f1_score, color='#8DA0CB', width=bar_width, label='F1-score')

#Customize the plot
plt.xlabel('Algorithms')
plt.ylabel('Score')
plt.title('Algorithm Performance')
plt.xticks([r + bar_width for r in range(len(algorithms))], algorithms)
plt.legend()

#Saving the graph as an image
plt.savefig('static/performance_graph.png')

#Closing the plot
plt.close()