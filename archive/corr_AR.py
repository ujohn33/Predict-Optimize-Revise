import numpy as np

# Estimate the model parameters
p = 2  # Assume the model is AR(2)
alpha = [0.8, 0.3]
sigma2 = 0.1

# Initialize the covariance matrix with the variance of the white noise term
n = 100  # Number of observations in the time series
C = np.eye(n) * sigma2

# Calculate the covariance between Z_t,k+j and Z_t,k for j > p
for j in range(p+1, n):
    for i in range(p):
        C[j, j-i-1] = alpha[i] * C[j-1, j-i-1]
        C[j-i-1, j] = C[j, j-i-1]

# Calculate the covariance between Z_t,k+j and Z_t,k for j <= p
for j in range(1, p+1):
    for i in range(j):
        C[j, j-i-1] = alpha[i] * C[j-1, j-i-1]
        C[j-i-1, j] = C[j, j-i-1]

