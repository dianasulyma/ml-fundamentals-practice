import numpy as np
import matplotlib.pyplot as plt

'''
Code for gradient descent was adapted from the source below:
Author: NeuralNine (YouTube)
Date: July 15, 2025
Licence: N/A
Title: Gradient Descent From Scratch in Python - Visual Explanation
Availability: https://www.youtube.com/watch?v=gsfbWn4Gy5Q&t=982s
'''
# Simplified function than in NeuralNine demo
def z_function(x, y): 
    return 0.5 * x**2 + 2 * y**2 # defines the slope of the function, which is a paraboloid in this case

def calculate_gradient(x, y):
    return x, 4 * y # calculates the gradient of the function at a given point (x, y)

# Create a grid of points for visualization
x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 1, 0.01)
X, Y = np.meshgrid(x, y)
Z = z_function(X, Y)         

# Initialize the starting point and learning rate
current_point = (0.9, 0.9, z_function(0.9, 0.9))
learning_rate = 0.1
path = [current_point]

# Perform gradient descent for a set number (50) of iterations
for _ in range(50):
    dx, dy = calculate_gradient(current_point[0], current_point[1])
    new_x = current_point[0] - learning_rate * dx
    new_y = current_point[1] - learning_rate * dy
    current_point = (new_x, new_y, z_function(new_x, new_y))
    path.append(current_point)

# Extract the x, y, and z coordinates from the path for plotting
px, py, pz = map(np.array, zip(*path))

# Create a 3D plot to visualize the gradient descent path
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

ax.computed_z_order = False                      
ax.plot_surface(X, Y, Z, cmap="plasma", alpha=0.55, zorder=1)
ax.plot(px, py, pz + 0.06, color="#00E5FF", lw=2.5,
        marker="o", ms=4, zorder=10)            
ax.scatter(px[0], py[0], pz[0] + 0.06, color="white", s=70, zorder=11)

ax.view_init(elev=25, azim=45)
plt.show()