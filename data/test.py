import numpy as np
import matplotlib.pyplot as plt

# Create a sample 2D numpy array with the specified shape
data = np.random.rand(5000, 2000)

# Transpose the array to have the data along the y-axis
data = data.T

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title("2D Array Plot")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()