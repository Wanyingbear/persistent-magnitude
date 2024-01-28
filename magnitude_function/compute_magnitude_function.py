# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:30:05 2023

@author: 13244
"""

import matplotlib.pyplot as plt
import numpy as np
import gudhi

# Read data
filename = 'B15C2H17-7.xyz'
# Array to store data
data_array = []

# Open the file and read its content
with open(filename, 'r') as file:
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


# Create a persistent diagram
rips_complex = gudhi.RipsComplex(points=data_array, max_edge_length=11)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
simplex_tree.compute_persistence()
# Extract endpoints of 0-dimensional barcode

# Build the magnitude function
t_values = np.linspace(0, 4, 100)  # Adjust the range as needed

# Initialize an array to store function values
function_values = np.zeros(len(t_values))

# Calculate the function value for each t
for i, t in enumerate(t_values):
    function_value = 0
    for interval in simplex_tree.persistence():      
        start_point, end_point = interval[1]
#        print(interval[0])
        if np.isscalar(end_point) and np.isinf(end_point):
            function_value = function_value + (-1)**(interval[0])*np.exp(-start_point * t)
        else:
            function_value = function_value + (-1)**(interval[0])*(np.exp(-start_point * t) - np.exp(-end_point * t))
    function_values[i] = function_value
    

# Create a plot

# Set the image's dpi to 300, and the size of the image to 3280 x 2780
plt.figure(figsize=(32.8, 27.8), dpi=300)

# Set the font size
plt.rcParams.update({'font.size': 120})

# Adjust the thickness of the horizontal and vertical axes and border lines
plt.axhline(linewidth=8, color='black')  # Horizontal axis
plt.axvline(linewidth=8, color='black')  # Vertical axis

# Plotting
plt.plot(t_values, function_values, linewidth=8)
plt.xlabel('t')
plt.ylabel('M(t)')
plt.title('Magnitude Function')
plt.grid(True)

plt.show()










