import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Define Gauss quadrature points and weights
def gauss1d(nqp):
    if nqp == 2:
        xi = torch.tensor([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        w = torch.tensor([1.0, 1.0])
    else:
        raise ValueError("Unsupported number of quadrature points")
    return xi, w


# Define shape functions and their derivatives for line elements
def shape1d(xi, nen):
    if nen == 2:
        N = torch.tensor([(1 - xi) / 2, (1 + xi) / 2])
        dN = torch.tensor([-0.5, 0.5])
    else:
        raise ValueError("Unsupported number of element nodes")
    return N, dN


# Calculate Jacobian for 1D elements
def jacobian1d(xe, dN):
    J = dN @ xe
    if J <= 0:
        raise ValueError(f"Invalid Jacobian determinant: {J}")
    return J, 1 / J


def create_annular_mesh(inner_radius, outer_radius, num_radial_divisions, num_circumferential_divisions):
    theta = np.linspace(0, 2 * np.pi, num_circumferential_divisions + 1)
    r = np.linspace(inner_radius, outer_radius, num_radial_divisions + 1)
    theta, r = np.meshgrid(theta, r)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def plot_annular_mesh(x, y):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    for i in range(x.shape[0]):
        for j in range(x.shape[1] - 1):
            ax.plot([x[i, j], x[i, (j + 1) % x.shape[1]]], [y[i, j], y[i, (j + 1) % x.shape[1]]], 'k-')
    for j in range(x.shape[1]):
        for i in range(x.shape[0] - 1):
            ax.plot([x[i, j], x[i + 1, j]], [y[i, j], y[i + 1, j]], 'k-')
    plt.show()


def analysis():
    inner_radius = 0.225  # meters
    outer_radius = 0.420  # meters
    num_radial_divisions = 10
    num_circumferential_divisions = 20

    x, y = create_annular_mesh(inner_radius, outer_radius, num_radial_divisions, num_circumferential_divisions)
    plot_annular_mesh(x, y)


# Call the analysis function
analysis()
