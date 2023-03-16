import plotly.graph_objects as go
import numpy as np
import random

# Define the objective function


# Define the gradient descent algorithm with batch gradient descent
def gradient_descent(nodes, radius, n_facilities, step_size=0.1, max_iter=1000, tol=1e-6):
    # Initialize the location of the facilities randomly
    x = np.zeros((n_facilities, len(nodes[0])))
    for i in range(n_facilities):
        x[i] = random.choice(nodes)

    # Iterate until convergence
    for i in range(max_iter):
        # Calculate the objective function and gradient for each facility
        f = np.zeros(n_facilities)
        grad = np.zeros((n_facilities, len(nodes[0])))
        for j in range(n_facilities):
            f[j] = objective_function(x, nodes, radius, j)
            grad[j] = gradient(x, nodes, radius, j)

        # Update the location of each facility
        x_new = np.zeros((n_facilities, len(nodes[0])))
        for j in range(n_facilities):
            x_new[j] = x[j] - step_size * np.mean(grad, axis=0)

        # Check for convergence
        f_new = np.zeros(n_facilities)
        for j in range(n_facilities):
            f_new[j] = objective_function(x_new, nodes, radius, j)
        if np.all(abs(f_new - f) < tol):
            break

        x = x_new

    return x

# Define the modified objective function
# Define the modified objective function with overlap penalty
def objective_function(x, nodes, radius, j, alpha=1.0, beta=1.0):
    covered = np.zeros(len(nodes))
    for i in range(len(x)):
        if i != j:  # Ignore the j-th facility
            for k in range(len(nodes)):
                if np.linalg.norm(x[i] - nodes[k]) <= radius:
                    covered[k] = 1
    for k in range(len(nodes)):
        if np.linalg.norm(x[j] - nodes[k]) <= radius:
            covered[k] = 1
    num_covered = np.sum(covered)

    overlap = 0
    for i in range(len(x)):
        if i != j:  # Ignore the j-th facility
            overlap += max(0, radius - np.linalg.norm(x[i] - x[j]))

    return alpha * num_covered - beta * overlap


# Define the modified gradient function
def gradient(x, nodes, radius, j):
    grad = np.zeros(len(nodes[0]))
    for i in range(len(x)):
        if i != j:  # Ignore the j-th facility
            for k in range(len(nodes)):
                if np.linalg.norm(x[i] - nodes[k]) <= radius:
                    grad += (x[i] - nodes[k])
    for k in range(len(nodes)):
        if np.linalg.norm(x[j] - nodes[k]) <= radius:
            grad += (x[j] - nodes[k])
    return grad


# Example usage
nodes = np.random.rand(50, 2)  # Generate random nodes
radius = 0.2
n_facilities = 5
optimal_location = gradient_descent(nodes, radius, n_facilities)

