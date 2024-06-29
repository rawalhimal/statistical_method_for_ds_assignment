# Import necessary libraries
import pandas as pd  # For handling data in DataFrames
import matplotlib.pyplot as plt  # For creating plots
import seaborn as sns  # For enhanced visualizations

# Load the dataset from CSV file
data = pd.read_csv('gene_data.csv')

# Define the gene columns
gene_columns = ['x1', 'x2', 'x3', 'x4', 'x5']

# --------------------------------------------------------------------------------
# Task 1: Preliminary Data Analysis
# --------------------------------------------------------------------------------

# Plot time series for each gene
fig, axes = plt.subplots(len(gene_columns), 1, figsize=(10, 12), sharex=True)

for ax, gene in zip(axes, gene_columns):
    ax.plot(data['Time'], data[gene], label=gene)
    ax.set_ylabel('Expression Level')
    ax.set_title(f'Time Series Plot for {gene}')
    ax.legend()

# Set the x-axis label for the bottom plot
axes[-1].set_xlabel('Time (minutes)')

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------
# Task 1: Distribution Analysis of Each Simulated Gene
# --------------------------------------------------------------------------------

# Create distribution plots for each gene
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

for i, gene in enumerate(gene_columns):
    row, col = divmod(i, 2)
    sns.histplot(data[gene], ax=axes[row, col], bins=30, kde=True, color='skyblue')
    sns.kdeplot(data[gene], ax=axes[row, col], color='red')
    axes[row, col].set_title(f'Distribution Plot for {gene}')
    axes[row, col].set_xlabel('Expression Level')
    axes[row, col].set_ylabel('Density')
    axes[row, col].legend(labels=['Histogram', 'Density Curve'])

# Hide the empty subplot in the grid
axes[2, 1].axis('off')

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------
# Task 1: Correlation and Scatter Plots to Examine Dependencies
# --------------------------------------------------------------------------------

# Compute the correlation matrix
correlation_matrix = data.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Generate scatter plots for pairwise gene comparisons
sns.pairplot(data[gene_columns])
plt.show()
