import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import uniform

# Function to perform linear regression using the normal equation
def perform_linear_regression(X, y):
    # Add bias term by appending a column of ones to X
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # Compute necessary matrices
    XtX = X_bias.T @ X_bias
    Xty = X_bias.T @ y
    
    # Solve for theta (including bias term)
    theta = np.linalg.solve(XtX, Xty)
    
    # Return the parameters theta
    return theta

#---------------------------------------------------
# TASK 2.1
#------------------------------------------------------

# Load the data from CSV
data = pd.read_csv('gene_data.csv')

# Extract input features and target variable
X = data[['x1', 'x3', 'x4', 'x5']].values
y = data['x2'].values

# Define different models
# Model 1: x2 = theta1*x4 + theta2*x3^2 + theta_bias
X_model1 = X[:, [2, 1]]   # Select relevant columns (x4, x3)
theta1 = perform_linear_regression(X_model1, y)
print(f"Parameters for Model 1: {theta1}")

# Model 2: x2 = theta1*x4 + theta2*x3^2 + theta3*x5 + theta_bias
X_model2 = X[:, [2, 1, 3]]   # Select relevant columns (x4, x3, x5)
theta2 = perform_linear_regression(X_model2, y)
print(f"Parameters for Model 2: {theta2}")

# Model 3: x2 = theta1*x3 + theta2*x4 + theta3*x5^3
X_model3 = X[:, [1, 2, 3]]   # Select relevant columns (x3, x4, x5)
theta3 = perform_linear_regression(X_model3, y)
print(f"Parameters for Model 3: {theta3}")

# Model 4: x2 = theta1*x4 + theta2*x3^2 + theta3*x5^3 + theta_bias
X_model4 = X[:, [2, 1, 3]]   # Select relevant columns (x4, x3, x5)
theta4 = perform_linear_regression(X_model4, y)
print(f"Parameters for Model 4: {theta4}")

# Model 5: x2 = theta1*x4 + theta2*x1^2 + theta3*x3^2 + theta_bias
X_model5 = X[:, [2, 0, 1]]   # Select relevant columns (x4, x1, x3)
theta5 = perform_linear_regression(X_model5, y)
print(f"Parameters for Model 5: {theta5}")

#---------------------------------------------------
# TASK 2.2
#------------------------------------------------------

# Function to calculate RSS (Residual Sum of Squares)
def calculate_rss(X, y, theta):
    # Add bias term by appending a column of ones to X
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # Predicted values
    y_pred = X_bias @ theta
    
    # Residuals
    residuals = y - y_pred
    
    # RSS (Residual Sum of Squares)
    rss = np.sum(residuals ** 2)
    
    return rss

# Calculate RSS for each model
rss1 = calculate_rss(X_model1, y, theta1)
rss2 = calculate_rss(X_model2, y, theta2)
rss3 = calculate_rss(X_model3, y, theta3)
rss4 = calculate_rss(X_model4, y, theta4)
rss5 = calculate_rss(X_model5, y, theta5)

# Print RSS for each model
print(f"RSS for Model 1: {rss1}")
print(f"RSS for Model 2: {rss2}")
print(f"RSS for Model 3: {rss3}")
print(f"RSS for Model 4: {rss4}")
print(f"RSS for Model 5: {rss5}")

#---------------------------------------------------
# TASK 2.3
#------------------------------------------------------

# Function to compute log-likelihood for a given model
def compute_log_likelihood(X, y, theta):
    # Add bias term by appending a column of ones to X
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # Predicted values
    y_pred = X_bias @ theta
    
    # Residuals
    residuals = y - y_pred
    
    # RSS (Residual Sum of Squares)
    rss = np.sum(residuals ** 2)
    
    # Number of observations
    n = len(X)
    
    # Degrees of freedom (number of observations minus number of parameters estimated)
    df = n - X_bias.shape[1]
    
    # Estimate variance of residuals
    sigma2 = rss / df
    
    # Calculate log-likelihood
    log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 / sigma2 * rss
    
    return log_likelihood

log_likelihood1 = compute_log_likelihood(X_model1, y, theta1)
log_likelihood2 = compute_log_likelihood(X_model2, y, theta2)
log_likelihood3 = compute_log_likelihood(X_model3, y, theta3)
log_likelihood4 = compute_log_likelihood(X_model4, y, theta4)
log_likelihood5 = compute_log_likelihood(X_model5, y, theta5)

print(f"Log-Likelihood for Model 1: {log_likelihood1}")
print(f"Log-Likelihood for Model 2: {log_likelihood2}")
print(f"Log-Likelihood for Model 3: {log_likelihood3}")
print(f"Log-Likelihood for Model 4: {log_likelihood4}")
print(f"Log-Likelihood for Model 5: {log_likelihood5}")

#--------------------------------------------------------
# TASK 2.4
#-------------------------------------------------------

# Function to calculate AIC for a given model
def calculate_aic(log_likelihood, num_params):
    return -2 * log_likelihood + 2 * num_params

# Function to calculate BIC for a given model
def calculate_bic(log_likelihood, num_params, n):
    return -2 * log_likelihood + num_params * np.log(n)

num_params1 = len(theta1)
num_params2 = len(theta2)
num_params3 = len(theta3)
num_params4 = len(theta4)
num_params5 = len(theta5)

# Calculate AIC for each model
aic1 = calculate_aic(log_likelihood1, num_params1)
aic2 = calculate_aic(log_likelihood2, num_params2)
aic3 = calculate_aic(log_likelihood3, num_params3)
aic4 = calculate_aic(log_likelihood4, num_params4)
aic5 = calculate_aic(log_likelihood5, num_params5)

# Calculate BIC for each model
n = len(X)
bic1 = calculate_bic(log_likelihood1, num_params1, n)
bic2 = calculate_bic(log_likelihood2, num_params2, n)
bic3 = calculate_bic(log_likelihood3, num_params3, n)
bic4 = calculate_bic(log_likelihood4, num_params4, n)
bic5 = calculate_bic(log_likelihood5, num_params5, n)

# Print AIC and BIC for each model
print(f"AIC for Model 1: {aic1}")
print(f"AIC for Model 2: {aic2}")
print(f"AIC for Model 3: {aic3}")
print(f"AIC for Model 4: {aic4}")
print(f"AIC for Model 5: {aic5}")

print(f"BIC for Model 1: {bic1}")
print(f"BIC for Model 2: {bic2}")
print(f"BIC for Model 3: {bic3}")
print(f"BIC for Model 4: {bic4}")
print(f"BIC for Model 5: {bic5}")

#--------------------------------------------------------
# TASK 2.5
#-------------------------------------------------------

# Function to compute residuals
def compute_residuals(X, y, theta):
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    y_pred = X_bias @ theta
    residuals = y - y_pred
    return residuals

# Function to plot histograms and Q-Q plots
def plot_residual_analysis(residuals, model_name):
    plt.figure(figsize=(12, 5))

    # Plot histogram
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.title(f'{model_name} Residuals Histogram')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')

    # Q-Q plot
    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{model_name} Residuals Q-Q Plot')

    plt.tight_layout()
    plt.show()

# Compute residuals for each model
residuals1 = compute_residuals(X_model1, y, theta1)
residuals2 = compute_residuals(X_model2, y, theta2)
residuals3 = compute_residuals(X_model3, y, theta3)
residuals4 = compute_residuals(X_model4, y, theta4)
residuals5 = compute_residuals(X_model5, y, theta5)

# Plot residuals analysis
plot_residual_analysis(residuals1, 'Model 1')
plot_residual_analysis(residuals2, 'Model 2')
plot_residual_analysis(residuals3, 'Model 3')
plot_residual_analysis(residuals4, 'Model 4')
plot_residual_analysis(residuals5, 'Model 5')
# TASK 2.6
#-------------------------------------------------------

# Create a dataframe containing model residuals, AIC, and BIC
data = {'ModelName': ['Model1', 'Model2', 'Model3', 'Model4', 'Model5'],
        'RSS': [rss1, rss2, rss3, rss4, rss5],
        'AIC': [aic1, aic2, aic3, aic4, aic5],
        'BIC': [bic1, bic2, bic3, bic4, bic5]}

df_RSS_AIC_BIC = pd.DataFrame(data)
print(df_RSS_AIC_BIC)

#--------------------------------------------------------
# TASK 2.7
#-------------------------------------------------------

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Prepare data for Model 5
X_train_model5 = np.column_stack((X_train[:, 2], X_train[:, 0]**2, X_train[:, 1]**2))
X_test_model5 = np.column_stack((X_test[:, 2], X_test[:, 0]**2, X_test[:, 1]**2))

# Estimate parameters using the training dataset
theta_model5 = compute_linear_regression(X_train_model5, y_train)
print("Estimated Parameters:", theta_model5)

# Compute predictions on the testing dataset
X_test_with_bias_model5 = np.column_stack((np.ones(len(X_test_model5)), X_test_model5))
y_pred_model5 = np.dot(X_test_with_bias_model5, theta_model5)
print('Output Prediction on Testing Dataset:', y_pred_model5)

# Calculate Residuals
residuals = y_test - y_pred_model5

# Calculate RSS
rss = np.sum(residuals ** 2)
print(f"Residual Sum of Squares (RSS): {rss}")

# Compute 95% confidence intervals using statsmodels
X_train_with_bias_model5 = np.column_stack((np.ones(len(X_train_model5)), X_train_model5))
model = sm.OLS(y_train, X_train_with_bias_model5).fit()
predictions = model.get_prediction(X_test_with_bias_model5)
prediction_summary_frame = predictions.summary_frame(alpha=0.05)

y_pred = prediction_summary_frame['mean']
conf_int_lower = prediction_summary_frame['obs_ci_lower']
conf_int_upper = prediction_summary_frame['obs_ci_upper']

# Create DataFrame for results
results_df = pd.DataFrame({
    'True Values': y_test,
    'Predictions': y_pred,
    'Conf Lower Bound': conf_int_lower,
    'Conf Upper Bound': conf_int_upper
})

# Display DataFrame
print(results_df)

# Plot the predictions, confidence intervals, and testing data
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='True Values', marker='o')
plt.plot(range(len(y_pred)), y_pred, label='Predictions', marker='x')
plt.fill_between(range(len(y_test)), conf_int_lower, conf_int_upper, color='gray', alpha=0.2, label='95% CI')
plt.xlabel('Test Sample Index')
plt.ylabel('Gene Expression Level')
plt.title('Model 5 Predictions with 95% Confidence Intervals')
plt.legend()
plt.show()

# Plot histogram and density plot of y_train (output data)
plt.figure(figsize=(10, 6))

# Histogram
sns.histplot(y_train, bins=20, kde=False, color='blue', label='Histogram', stat='density')

# Density plot
sns.kdeplot(y_train, color='red', label='Density Plot')

plt.xlabel('Gene Expression Level (y)')
plt.ylabel('Density')
plt.title('Distribution of Output Data (y_train)')
plt.legend()
plt.show()

#--------------------------------------------------------
# TASK 3
#-------------------------------------------------------

# Estimated parameters from Task 2.1
theta_bias = 1.70512167
theta1 = 0.57269652
theta2 = -0.25593985
theta3 = -1.08848322

# Define ranges for the uniform prior around the estimated values
prior_ranges = {
    'theta_bias': (theta_bias - 1, theta_bias + 1),
    'theta1': (theta1 - 1, theta1 + 1)
}

# Function to simulate the model output based on given parameters
def simulate_model(theta_bias, theta1, theta2, theta3, X):
    return (theta_bias + theta1 * X[:, 0] + theta2 * (X[:, 1] ** 2) + theta3 * (X[:, 2] ** 2))

# Perform Rejection ABC
accepted_samples = []
epsilon = 100.0  # Set a reasonable threshold for RSS
num_samples = 1000
max_iterations = 100000  # Maximum iterations to avoid infinite loop
iteration = 0

while len(accepted_samples) < num_samples and iteration < max_iterations:
    theta_bias_sample = np.random.uniform(prior_ranges['theta_bias'][0], prior_ranges['theta_bias'][1])
    theta1_sample = np.random.uniform(prior_ranges['theta1'][0], prior_ranges['theta1'][1])
    
    model_output = simulate_model(theta_bias_sample, theta1_sample, theta2, theta3, X_train)
    
    # Calculate RSS
    rss = np.sum((model_output - y_train) ** 2)
    
    # Rejection step
    if rss < epsilon:
        accepted_samples.append((theta_bias_sample, theta1_sample, rss))
    
    iteration += 1

# Convert accepted samples to a DataFrame
accepted_samples_df = pd.DataFrame(accepted_samples, columns=['Theta_bias', 'Theta1', 'RSS'])

# Plot joint and marginal posterior distributions
if not accepted_samples_df.empty:
    sns.pairplot(accepted_samples_df[['Theta_bias', 'Theta1']], kind='kde')
    plt.suptitle('Joint Posterior Distributions', y=1.02)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(accepted_samples_df['Theta_bias'], kde=True, ax=axes[0])
    axes[0].set_xlabel('Theta_bias')
    axes[0].set_title('Marginal Posterior Distribution of Theta_bias')

    sns.histplot(accepted_samples_df['Theta1'], kde=True, ax=axes[1])
    axes[1].set_xlabel('Theta1')
    axes[1].set_title('Marginal Posterior Distribution of Theta1')

    plt.tight_layout()
    plt.show()

    # Scatter plot of RSS
    plt.figure(figsize=(10, 6))
    plt.scatter(accepted_samples_df['Theta_bias'], accepted_samples_df['Theta1'], c=accepted_samples_df['RSS'], cmap='viridis', marker='o')
    plt.colorbar(label='RSS')
    plt.xlabel('Theta_bias')
    plt.ylabel('Theta1')
    plt.title('Scatter Plot of Accepted Samples with RSS')
    plt.show()
else:
    print("No samples were accepted. Consider increasing the tolerance level further.")