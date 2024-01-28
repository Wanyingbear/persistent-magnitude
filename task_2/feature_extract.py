# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 12:55:41 2023

@author: Lenovo
"""

import numpy as np
import gudhi
import os
import re


# Specify the path for the new text file
output_file_path = 'magnitude_feature.txt'

# Initialize an empty list to store vectors
features = []

# Specify folder path
folder_path = 'structures_BCH'

# Define a function to extract numeric values from the filename
def extract_numbers(filename):
    numbers = re.findall(r'\d+', filename)
    return tuple(map(int, numbers))

# Get a sorted list of filenames based on numeric values
file_list = sorted(os.listdir(folder_path), key=extract_numbers)

count = 0

# Iterate through sorted filenames
for filename in file_list:
    # Construct the full file path
    file_path = os.path.join(folder_path, filename)

    # Use the rest of your code to process the file
    data_array = []
    with open(file_path, 'r') as file:
        file.readline()
        file.readline()
        for line in file:
            row_data = line.split()
            data = [float(value) for value in row_data[1:]]
            data_array.append(data)

    atoms_num = len(data_array)

    # Create a persistent diagram
    rips_complex = gudhi.RipsComplex(points=data_array, max_edge_length=11)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    simplex_tree.compute_persistence()

    t_values = np.linspace(0, 4, 100)  # Adjust the range as needed

    # Initialize an array to store function values
    function_values = np.zeros(len(t_values))

    # Calculate the function value for each t
    for i, t in enumerate(t_values):
        function_value = 0
        for interval in simplex_tree.persistence():
            start_point, end_point = interval[1]
            if np.isscalar(end_point) and np.isinf(end_point):
                function_value = function_value + (-1)**(interval[0])*np.exp(-start_point * t)
            else:
                function_value = function_value + (-1)**(interval[0])*(np.exp(-start_point * t) - np.exp(-end_point * t))
        function_values[i] = function_value

    count = count + 1
    print(count)
    features.append(function_values)

# Convert the list to a NumPy array
features_array = np.array(features)

# Use numpy.savetxt to write the array to a file
np.savetxt(output_file_path, features_array)


        
        
        
        
        
        
        
        
        
        
        
        
        