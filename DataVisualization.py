# Data Visualiztion. 

#Import the needed package to run the code. 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set the pathway where you have saved the file with the data.
data = pd.read_csv('/Users/rodrigosanchez/Documents/ScientificProgramming/Experiment.csv')

# Taking the two features and measuring the accuracy. 
accuracy_data = data.groupby(['ContrastHeterogeneity', 'GridCoarseness']).mean()['Correct'].unstack() 

# Plotting. 
plt.figure(figsize=(10, 8))
sns.heatmap(accuracy_data, cmap="coolwarm", vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'}, annot=True)
plt.title("Contrast Heterogeneity vs Grid Coarseness")
plt.xlabel("Grid Coarseness")
plt.ylabel("Contrast Heterogeneity")
plt.show()

# Calculate the mean accuracy by the feature “Condition”. 
accuracy_by_condotion = data.groupby('Condition').mean()['Correct']

# Plotting. 
plt.figure(figsize=(12, 6))
accuracy_by_condotion.plot(kind='bar', color='skyblue')
plt.title('Response Accuracy by Condition')
plt.xlabel('Condition')
plt.ylabel('Accuracy')
plt.show()

# Take the data from the “ContrastHeterogeneity” and “GridCoarseness” columns, making combinations and measure the 
# mean of the “Correct” column based on the combinations of these variables. 
accuracy_by_grid_and_contrast = data.groupby(['GridCoarseness', 'ContrastHeterogeneity']).mean(['Correct']).reset_index()

# Plotting. 
plt.figure(figsize=(10, 6))
sns.lineplot(x='GridCoarseness', y='Correct', hue='ContrastHeterogeneity', marker='o', data=accuracy_by_grid_and_contrast)
plt.title('Accuracy by Grid Coarseness for different Contrast Heterogeneity')
plt.xlabel('Grid Coarseness')
plt.ylabel('Accuracy')
plt.legend(title='Contrast Heterogeneity')
plt.show()
