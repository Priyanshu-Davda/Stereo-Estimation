import pandas as pd
import numpy as np

# Define the checkerboard corner dimensions
rows = 5  # Number of corners in the row
cols = 8  # Number of corners in the column
square_size = 30  # Square size in mm (assuming each square is 30mm)

# Generate the 3D coordinates for the checkerboard
points = []

for i in range(rows):
    for j in range(cols):
        # The x and y coordinates are based on the square size and row/column position
        x = j * square_size
        y = i * square_size
        z = 0  # All points lie on the same plane, so z = 0
        point_name = f"p{(i * cols) + j + 1}"  # Point name as p1, p2, ..., p40
        points.append([point_name, x, y, z])

# Create a DataFrame with the specified format
df = pd.DataFrame(points, columns=["Pointname", "x", "y", "z"])

# Save the DataFrame to an Excel file
df.to_excel("checkerboard_3d_coordinates.xlsx", index=False)
