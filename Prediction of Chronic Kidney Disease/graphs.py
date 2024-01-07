import matplotlib.pyplot as plt
import seaborn as sns

# Accuracy data
algorithms = ['KNN', 'Decision Tree']
accuracy = [80, 86.25]

# Set the style using seaborn
sns.set(style='darkgrid')

# Define colors for the bars
colors = ['#66C2A5', '#FC8D62']

# Plotting the graph
plt.bar(algorithms, accuracy, color=colors)

# Add percentage labels on top of bars
for i, acc in enumerate(accuracy):
    plt.text(i, acc + 1, f'{acc}%', ha='center', color='black')

# Customize the plot
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')

# Saving the graph as an image
plt.savefig('static/accuracy_graph1.png')

# Closing the plot
plt.close()
