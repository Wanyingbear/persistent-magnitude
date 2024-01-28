# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:40:18 2023

@author: Lenovo
"""

import numpy as np
import gudhi
import os
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

features = []
# Specify folder path
folder_path = 'all_stru'

# Traverse all files in the folder
for foldername, subfolders, filenames in os.walk(folder_path):

    for filename in filenames:
        # Construct the full file path
        file_path = os.path.join(foldername, filename)

        # Array to store data
        data_array = []

        # Open the file and read its content
        with open(file_path, 'r') as file:
            # Skip the first two lines
            file.readline()
            file.readline()

            # Read data line by line
            for line in file:
                # Use the split() method to split the line by space or tab
                row_data = line.split()

                # Extract data starting from the second column and convert to the appropriate data type
                data = [float(value) for value in row_data[1:]]

                # Add data to the array
                data_array.append(data)

        # Print the result
        # print(filename, ":", data_array)

        atoms_num = len(data_array)

        # Create a persistent diagram
        rips_complex = gudhi.RipsComplex(points=data_array, max_edge_length=11)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
        simplex_tree.compute_persistence()

        # Get Betti curve data
        betti_numbers = simplex_tree.betti_numbers()

        # Get 0-dimensional average bar length
        sum = 0
        num = 0
        for interval in simplex_tree.persistence():
            if interval[0] == 0:
                bar = interval[1]
                if np.isscalar(bar[1]) and np.isinf(bar[1]):
                    continue
                else:
                    sum = sum + bar[1]
                    num = num +1

        average_bar_length = sum/(num)

        # Extract and store features
        one_feature = [average_bar_length/atoms_num]
        features.append(one_feature)


features_array = np.array(features)

relative_energy = np.array([
    -4.566912632,
    -3.25692631,
    -3.614181583,
    -3.381404523,
    -3.500432844,
    -4.072963135,
    -3.696135953,
    -3.811412098,
    -3.824863253,
    0,
    -0.393185344,
    -1.214913058,
    -1.282469257,
    -1.948727467,
    -3.391379355,
    -1.462995199
])

num_rows = features_array.shape[0]

# List to store predicted values
predicted_values_list = []

for i in range(num_rows):
    # Remove the nth row from the array
    reduced_matrix = np.delete(features_array, i, axis=0)

    # Remove the nth element from the list
    reduced_list = np.delete(relative_energy, i)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(reduced_matrix, reduced_list)

    # Predict the nth row of the array
    predicted_value = model.predict([features_array[i]])

    # Add the predicted value to the list
    predicted_values_list.append(predicted_value[0])

    # Print the result
    #    print(f"For i={i}: Predicted value: {predicted_value}, Actual value: {relative_energy[i]}")

# Print the list of predicted values
# print("Predicted values list:", predicted_values_list)

# Calculate Pearson correlation coefficient
correlation_coefficient, _ = pearsonr(relative_energy, predicted_values_list)

# Calculate root mean squared error (RMSE)
rmse = np.sqrt(mean_squared_error(relative_energy, predicted_values_list))

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(relative_energy, predicted_values_list)

# Print the results
print("Pearson correlation coefficient:", correlation_coefficient)
print("RMSE:", rmse)
print("MAE:", mae)
