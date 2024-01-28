# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 13:56:36 2023

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import gudhi
import sympy as sp
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
                # Use split() method to split lines by space or tab
                row_data = line.split()

                # Extract data starting from the second column and convert to the appropriate data type
                data = [float(value) for value in row_data[1:]]

                # Add data to the array
                data_array.append(data) 

        # Print the result
#        print(filename, ":", data_array)
        
        atoms_num = len(data_array)

        # Create a persistent diagram
        rips_complex = gudhi.RipsComplex(points=data_array, max_edge_length=11)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
        simplex_tree.compute_persistence()
        # Extract endpoints of 0-dimensional barcode
        zero_dim_barcodes = []
        for interval in simplex_tree.persistence():
            if interval[0] == 0:  # 0 represents 0-dimensional
                if interval[1] == float('inf'):
                    # Infinite persistence
                    start_point = interval[0]
                    zero_dim_barcodes.append((start_point, float('inf')))
                else:
                    # Finite persistence
                    start_point = interval[0]
                    end_point = interval[1]
                    zero_dim_barcodes.append((start_point, end_point))
        # print("0-dimensional Barcode:")
        # print(zero_dim_barcodes)
        barcode_data = zero_dim_barcodes
        # Extract intervals
        intervals = [interval[1] for interval in barcode_data]
        # print(intervals)
        
        t_values = np.linspace(0, 4, 100)  # Adjust the range as needed
        
        # Initialize an array to store function values
        function_values = np.zeros(len(t_values))
        # Calculate the function value for each t
        for i, t in enumerate(t_values):
            function_value = 0
            for interval in intervals:
                start_point, end_point = interval
                if np.isscalar(end_point) and np.isinf(end_point):
                    function_value += np.exp(-start_point * t)
                else:
                    function_value += np.exp(-start_point * t) - np.exp(-end_point * t)
            function_values[i] = function_value
        

        # Define symbolic variable
        t = sp.symbols('t')
        
        # Initialize a symbolic variable to store function values
        function_value_sym = 0
        
        # Calculate the symbolic expression for the function value for each t
        for interval in intervals:
            start_point, end_point = interval
            if np.isscalar(end_point) and np.isinf(end_point):
                function_value_sym += sp.exp(-start_point * t)
            else:
                function_value_sym += sp.exp(-start_point * t) - sp.exp(-end_point * t)
                    
        # Use lambdify to convert the SymPy expression into a NumPy function
        function_values_sym = sp.lambdify(t, function_value_sym, 'numpy')
        
        # Calculate the numeric values of the function at t_values
        function_values_numeric = function_values_sym(t_values)
        
        # Calculate the derivative (slope)
        x_prime = sp.diff(t, t)
        y_prime = sp.diff(function_value_sym, t)
        
        
        slope_at_t0 = y_prime.subs(t, 0)
#        print("Slope at t=0:", slope_at_t0)
        
        value_t0 = function_values_sym(0)

  
        
        difference2_array = np.diff(function_values)
        index = np.argmax(difference2_array < 0.05)
        index_value = index/(25*atoms_num)

        one_feature=[slope_at_t0/atoms_num]  #Magnitude curve slope at 0, M’(0)/N;
        
#        one_feature=[value_t0/atoms_num]  #Magnitude curve value at 0, M(0)/N
 
#        one_feature=[index/atoms_num]  #Position where the slope first drops below 0.05, 0.8829201610993938

       
        features.append(one_feature)
        
#print(features)

features_array = np.array(features)

#calculated relative energy
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
print("Predicted values list:", predicted_values_list)
    
    
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
  
