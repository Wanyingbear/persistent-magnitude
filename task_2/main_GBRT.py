# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 23:02:27 2023

@author: Lenovo
"""


from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import time

# Record start time
start_time = time.time()

# Specify the file path containing array data
input_file_path = 'magnitude_feature.txt'

# Use numpy.loadtxt to load array data from the file
loaded_features = np.loadtxt(input_file_path)

# Specify Excel file path
excel_file_path = 'label_BCH.xlsx'

# Use pandas to read the Excel spreadsheet
known_data = pd.read_excel(excel_file_path, skiprows=0, usecols=[3, 4, 5])

# Convert data to NumPy array
calculated_values = known_data.to_numpy()

# Substitute Gradient Boosting Regressor model
X = loaded_features
# y = calculated_values[:,0]  # HOMO
y = calculated_values[:,1]  # LUMO
# y = calculated_values[:,2]  # HOMO–LUMO gap

# Create a Gradient Boosting Regressor model and set parameters
model = GradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=7,
    min_samples_leaf=1,
    min_samples_split=5,
    subsample=0.4,
    n_estimators=100
)

# Define 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validated predictions
y_pred = cross_val_predict(model, X, y, cv=kfold)

# Compute Pearson correlation coefficient
pearson_corr, _ = pearsonr(y, y_pred)
print(f'Pearson Correlation Coefficient: {pearson_corr}')

# Compute RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Compute MAE (Mean Absolute Error)
mae = mean_absolute_error(y, y_pred)
print(f'Mean Absolute Error: {mae}')

# Print each fold's performance
for i, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
    fold_y_true = y[test_idx]
    fold_y_pred = y_pred[test_idx]
    fold_rmse = np.sqrt(mean_squared_error(fold_y_true, fold_y_pred))
    fold_mae = mean_absolute_error(fold_y_true, fold_y_pred)
    print(f'Fold {i}: RMSE = {fold_rmse}, MAE = {fold_mae}')

# Print average performance
print(f'Average RMSE: {rmse}')
print(f'Average MAE: {mae}')

# Record end time
end_time = time.time()
# Calculate runtime
elapsed_time = end_time - start_time
# Print results
print(f"Elapsed Time: {elapsed_time} seconds")


































