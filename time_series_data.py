#Import all libraries

import pandas as pd #Pandas is used to store data in dataframe
import matplotlib.pyplot as plt #Matplotlib is used to plot figure
import seaborn as sns #Seaborn is used to make figure more interactive

# The CSV Files are provided where gene_data.csv contains 5 input simulated gene data
# X = {x1, x2, x3, x4} and time in minutes contains the sampling time of all simulated gene.
#--------------------------------------------------------------------------------
#Import data (gene_data.csv)
data = pd.read_csv('gene_data.csv')

#--------------------------------------------------------------------------------
#Task 1: Preliminary data analysis
#--------------------------------------------------------------------------------

#Time series plots (of each gene data)
genes = ['x1', 'x2', 'x3', 'x4', 'x5']
fig, axes = plt.subplots(len(genes), 1, figsize=(10, 12), sharex=True)

for ax, gene in zip(axes, genes):
    ax.plot(data['Time'], data[gene], label=gene)
    ax.set_ylabel('Expression Level')
    ax.set_title(f'Time Series Plot for {gene}')
    ax.legend()

# Set the x-axis label only for the bottom subplot
axes[-1].set_xlabel('Time (minutes)')

plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------
#Task 1: Distribution of Each Simulated gene data
#--------------------------------------------------------------------------------


fig, axes = plt.subplots(3, 2, figsize=(15, 15))

for i, gene in enumerate(genes):
    row = i // 2
    col = i % 2
    sns.histplot(data[gene], ax=axes[row, col], bins=30, kde=True, color='skyblue')
    sns.kdeplot(data[gene], ax=axes[row, col], color='red')
    axes[row, col].set_title(f'Distribution Plot for {gene}')
    axes[row, col].set_xlabel('Expression Level')
    axes[row, col].set_ylabel('Density')
    axes[row, col].legend(labels=['Histogram', 'Density Curve'])

# Hide the empty subplot
axes[2, 1].axis('off')

plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------
#Task 1: Correlation and scatter plots to examine their dependencies 
#--------------------------------------------------------------------------------


# Calculate correlation matrix
corr_matrix = data.corr()

# Display correlation matrix
print("Correlation Matrix:")
print(corr_matrix)


# Scatter plots
sns.pairplot(data[genes])
plt.show()

