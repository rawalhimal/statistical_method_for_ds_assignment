import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import uniform

# Function to perform linear regression using the normal equation
def linear_regression_normal_equation(X, y):
    # Add bias term by appending a column of ones to X
    X_with_bias = np.column_stack((np.ones(len(X)), X))
    
    # Compute necessary matrices
    X_transpose = np.transpose(X_with_bias)
    X_transpose_X = np.dot(X_transpose, X_with_bias)
    X_transpose_y = np.dot(X_transpose, y)
    
    # Solve for theta (including bias term)
    theta = np.linalg.solve(X_transpose_X, X_transpose_y)
    
    # Return the parameters theta
    return theta

#---------------------------------------------------
# TASK 2.1
#------------------------------------------------------

# Example usage:
# Load data using pandas
df = pd.read_csv('gene_data.csv')

# Extract columns into NumPy arrays
X_train = df[['x1', 'x3', 'x4', 'x5']].values
y_train = df['x2'].values

# Model1: x2 = theta1*x4 + theta2*x3^2 + theta_bias
X_model1 = X_train[:, [2, 1]]   # Adjust columns based on model equation (x4, x3)
theta_model1 = linear_regression_normal_equation(X_model1, y_train)
print(f"Parameters for Model1: {theta_model1}")

# Model2: x2 = theta1*x4 + theta2*x3^2 + theta3*x5 + theta_bias
X_model2 = X_train[:, [2, 1, 3]]   # Adjust columns based on model equation (x4, x3, x5)
theta_model2 = linear_regression_normal_equation(X_model2, y_train)
print(f"Parameters for Model2: {theta_model2}")

# Model3: x2 = theta1*x3 + theta2*x4 + theta3*x5^3
X_model3 = X_train[:, [1, 2, 3]]   # Adjust columns based on model equation (x3, x4, x5)
theta_model3 = linear_regression_normal_equation(X_model3, y_train)
print(f"Parameters for Model3: {theta_model3}")

# Model4: x2 = theta1*x4 + theta2*x3^2 + theta3*x5^3 + theta_bias
X_model4 = X_train[:, [2, 1, 3]]   # Adjust columns based on model equation (x4, x3, x5)
theta_model4 = linear_regression_normal_equation(X_model4, y_train)
print(f"Parameters for Model4: {theta_model4}")

# Model5: x2 = theta1*x4 + theta2*x1^2 + theta3*x3^2 + theta_bias
X_model5 = X_train[:, [2, 0, 1]]   # Adjust columns based on model equation (x4, x1, x3)
theta_model5 = linear_regression_normal_equation(X_model5, y_train)
print(f"Parameters for Model5: {theta_model5}")

#---------------------------------------------------
# TASK 2.2
#------------------------------------------------------

# Function to calculate RSS (Residual Sum of Squares)
def calculate_rss(X, y, theta):
    # Add bias term by appending a column of ones to X
    X_with_bias = np.column_stack((np.ones(len(X)), X))
    
    # Predicted values
    y_pred = np.dot(X_with_bias, theta)
    
    # Residuals
    residuals = y - y_pred
    
    # RSS (Residual Sum of Squares)
    rss = np.sum(residuals ** 2)
    
    return rss

# Calculate RSS for each model
rss_model1 = calculate_rss(X_model1, y_train, theta_model1)
rss_model2 = calculate_rss(X_model2, y_train, theta_model2)
rss_model3 = calculate_rss(X_model3, y_train, theta_model3)
rss_model4 = calculate_rss(X_model4, y_train, theta_model4)
rss_model5 = calculate_rss(X_model5, y_train, theta_model5)

# Print RSS for each model
print(f"RSS for Model1: {rss_model1}")
print(f"RSS for Model2: {rss_model2}")
print(f"RSS for Model3: {rss_model3}")
print(f"RSS for Model4: {rss_model4}")
print(f"RSS for Model5: {rss_model5}")

#---------------------------------------------------
# TASK 2.3
#------------------------------------------------------

# Function to compute log-likelihood for a given model
def compute_log_likelihood(X, y, theta):
    # Add bias term by appending a column of ones to X
    X_with_bias = np.column_stack((np.ones(len(X)), X))
    
    # Predicted values
    y_pred = np.dot(X_with_bias, theta)
    
    # Residuals
    residuals = y - y_pred
    
    # RSS (Residual Sum of Squares)
    rss = np.sum(residuals ** 2)
    
    # Number of observations
    n = len(X)
    
    # Degrees of freedom (number of observations minus number of parameters estimated)
    df = n - (X.shape[1] + 1)
    
    # Estimate variance of residuals
    sigma2 = rss / df
    
    # Calculate log-likelihood
    log_likelihood = -n/2 * np.log(2 * np.pi * sigma2) - 1/(2 * sigma2) * np.sum(residuals ** 2)
    
    return log_likelihood

log_likelihood_model1 = compute_log_likelihood(X_model1, y_train, theta_model1)
log_likelihood_model2 = compute_log_likelihood(X_model2, y_train, theta_model2)
log_likelihood_model3 = compute_log_likelihood(X_model3, y_train, theta_model3)
log_likelihood_model4 = compute_log_likelihood(X_model4, y_train, theta_model4)
log_likelihood_model5 = compute_log_likelihood(X_model5, y_train, theta_model5)

print(f"log_likelihood for Model1: {log_likelihood_model1}")
print(f"log_likelihood for Model2: {log_likelihood_model2}")
print(f"log_likelihood for Model3: {log_likelihood_model3}")
print(f"log_likelihood for Model4: {log_likelihood_model4}")
print(f"log_likelihood for Model5: {log_likelihood_model5}")

#--------------------------------------------------------
#TASK 2.4
#-------------------------------------------------------
# Function to calculate AIC for a given model
def calculate_aic(log_likelihood, num_params):
    return -2 * log_likelihood + 2 * num_params

# Function to calculate BIC for a given model
def calculate_bic(log_likelihood, num_params, n):
    return -2 * log_likelihood + num_params * np.log(n)


num_params_model1 = len(theta_model1)
num_params_model2 = len(theta_model2)
num_params_model3 = len(theta_model3)
num_params_model4 = len(theta_model4)
num_params_model5 = len(theta_model5)

# Calculate AIC for each model
aic_model1 = calculate_aic(log_likelihood_model1, num_params_model1)
aic_model2 = calculate_aic(log_likelihood_model2, num_params_model2)
aic_model3 = calculate_aic(log_likelihood_model3, num_params_model3)
aic_model4 = calculate_aic(log_likelihood_model4, num_params_model4)
aic_model5 = calculate_aic(log_likelihood_model5, num_params_model5)

# Calculate BIC for each model
n = len(X_train)
bic_model1 = calculate_bic(log_likelihood_model1, num_params_model1, n)
bic_model2 = calculate_bic(log_likelihood_model2, num_params_model2, n)
bic_model3 = calculate_bic(log_likelihood_model3, num_params_model3, n)
bic_model4 = calculate_bic(log_likelihood_model4, num_params_model4, n)
bic_model5 = calculate_bic(log_likelihood_model5, num_params_model5, n)


# Print AIC and BIC for each model
print(f"AIC for Model1: {aic_model1}")
print(f"AIC for Model2: {aic_model2}")
print(f"AIC for Model3: {aic_model3}")
print(f"AIC for Model4: {aic_model4}")
print(f"AIC for Model5: {aic_model5}")


print(f"BIC for Model1: {bic_model1}")
print(f"BIC for Model2: {bic_model2}")
print(f"BIC for Model3: {bic_model3}")
print(f"BIC for Model4: {bic_model4}")
print(f"BIC for Model5: {bic_model5}")

#--------------------------------------------------------
#TASK 2.5
#-------------------------------------------------------
# Function to compute residuals
def compute_residuals(X, y, theta):
    X_with_bias = np.column_stack((np.ones(len(X)), X))
    y_pred = np.dot(X_with_bias, theta)
    residuals = y - y_pred
    return residuals

# Function to plot histograms and Q-Q plots
def plot_error_distribution(residuals, model_name):
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

residuals_model1 = compute_residuals(X_model1, y_train, theta_model1)
plot_error_distribution(residuals_model1, "Model1")

residuals_model2 = compute_residuals(X_model2, y_train, theta_model2)
plot_error_distribution(residuals_model2, "Model2")

residuals_model3 = compute_residuals(X_model3, y_train, theta_model3)
plot_error_distribution(residuals_model3, "Model3")

residuals_model4 = compute_residuals(X_model4, y_train, theta_model4)
plot_error_distribution(residuals_model4, "Model4")

residuals_model5 = compute_residuals(X_model5, y_train, theta_model5)
plot_error_distribution(residuals_model5, "Model5")

#--------------------------------------------------------
#TASK 2.6
#-------------------------------------------------------

#Create a dataframe containing model residulas , AIC , BIC to select best model
data = {'ModelName':['Model1','Model2','Model3','Model4','Model5'],
        'RSS':[rss_model1,rss_model2,rss_model3,rss_model4,rss_model5],
        'AIC':[aic_model1,aic_model2,aic_model3,aic_model4,aic_model5],
        'BIC':[bic_model1,bic_model2,bic_model3,bic_model4,bic_model5]}
df_RSS_AIC_BIC = pd.DataFrame(data=data)
print(df_RSS_AIC_BIC)

#--------------------------------------------------------
#TASK 2.7
#-------------------------------------------------------

X = X_train
y = y_train

# Split the dataset (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Prepare the data for Model 5
# X_model5 = theta1 * x4 + theta2 * x1^2 + theta3 * x3^2 + theta_bias
X_train_model5 = np.column_stack((X_train[:, 2], X_train[:, 0]**2, X_train[:, 1]**2))
X_test_model5 = np.column_stack((X_test[:, 2], X_test[:, 0]**2, X_test[:, 1]**2))

# Function to estimate model parameters using the normal equation
def linear_regression_normal_equation(X, y):
    X_with_bias = np.column_stack((np.ones(len(X)), X))
    theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    return theta

# Estimate model parameters using the training dataset
theta_model5 = linear_regression_normal_equation(X_train_model5, y_train)
print("Estimated Parameters:", theta_model5)

# Compute model predictions on the testing dataset
X_test_with_bias_model5 = np.column_stack((np.ones(len(X_test_model5)), X_test_model5))
y_pred_model5 = np.dot(X_test_with_bias_model5, theta_model5)
print('Output Prediction on Testing Dataset is',y_pred_model5)

# Calculate Residuals
residuals = y_test - y_pred_model5

# Calculate RSS
RSS = np.sum(residuals ** 2)
print(f"Residual Sum of Squares (RSS): {RSS}")

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
#TASK 3
#-------------------------------------------------------


# Step 1: Assume you have estimated parameters from Task 2.1
# Here Task 2.1 values for Model5 [ 0.57269652 -1.08848322  1.70512167 -0.25593985]
theta_bias = 1.70512167
theta1 = 0.57269652
theta2 = -0.25593985
theta3 = -1.08848322


# Example values from Task 2.1
theta = [theta_bias, theta1, theta2, theta3]  # Add other parameters if necessary

# Define ranges for the uniform prior around the estimated values
prior_ranges = {
    'theta_bias': (theta_bias - 1, theta_bias + 1),
    'theta1': (theta1 - 1, theta1 + 1)
}

def simulate_model(theta_bias, theta1, theta2, theta3, X):
    """
    Simulates the model output based on given parameters.

    Parameters:
    - theta_bias: Bias parameter to be estimated.
    - theta1: First parameter to be estimated.
    - theta2: Fixed parameter.
    - theta3: Fixed parameter.
    - X: Input data matrix.

    Returns:
    - Model output: Predicted values based on the model.
    """
    model_output = (theta_bias
                    + theta1 * X[:, 0]  # x1
                    + theta2 * (X[:, 1] ** 2)  # x2^2
                    + theta3 * (X[:, 2] ** 2))  # x3^2
    return model_output


# Perform Rejection ABC
accepted_samples = []
tolerance = 0.1
num_samples = 1000
max_iterations = 100000  # Maximum iterations to avoid infinite loop
iteration = 0

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