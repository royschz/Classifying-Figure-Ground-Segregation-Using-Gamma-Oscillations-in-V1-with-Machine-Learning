#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:30:37 2024

@author: rodrigosanchez
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = data = pd.read_csv('/Users/rodrigosanchez/Documents/ScientificProgramming/Experiment.csv')
accuracy_data = data.groupby(['ContrastHeterogeneity', 'GridCoarseness']).mean()['Correct'].unstack()

plt.figure(figsize=(10, 8))
sns.heatmap(accuracy_data, annot=True, cmap="YlGnBu", cbar={'label': 'Accuracy'}, fmt=".2f")
plt.title("Accuracy Heatmap: Contrast Heterogeneity vs Grid Coarseness")
plt.xlabel("Grid Coarseness")
plt.ylabel("Contrast Heterogeneity")
plt.show()

#%%
accuracy_by_condotion = data.groupby('Condition').mean()['Correct']

plt.figure(figsize=(12, 6))

accuracy_by_condotion.plot(kind='bar', color='skyblue')
plt.title('Response Accuracy by Condition')
plt.xlabel('Condition')
plt.ylabel('Accuracy')

plt.show()

#%%

accuracy_by_grid_and_contrast = data.groupby(['GridCoarseness', 'ContrastHeterogeneity']).mean(['Correct']).reset_index()

plt.figure(figsize=(10, 6))

sns.lineplot(x='GridCoarseness', y='Correct', hue='ContrastHeterogeneity', marker='o', data=accuracy_by_grid_and_contrast)

plt.title('Accuracy by Grid Coarseness for different Contrast Heterogeneity')
plt.xlabel('Grid Coarseness')
plt.ylabel('Accuracy')
plt.legend(title='Contrast Heterogeneity')
plt.show()

#%% Posible version of the Arnold tongue 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('/Users/rodrigosanchez/Documents/ScientificProgramming/Experiment.csv')

accuracy_data = data.groupby(['ContrastHeterogeneity', 'GridCoarseness']).mean()['Correct'].unstack()

plt.figure(figsize=(10, 8))
sns.heatmap(accuracy_data, cmap="coolwarm", vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'}, annot=True)
plt.title("Arnold Tongue-like Plot: Contrast Heterogeneity vs Grid Coarseness")
plt.xlabel("Grid Coarseness")
plt.ylabel("Contrast Heterogeneity")
plt.show()




















