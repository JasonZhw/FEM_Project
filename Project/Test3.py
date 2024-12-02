import numpy as np
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

# Shape functions and derivatives for 8-node quadrilateral element
def shape_functions(xi, eta):
    N1 = 0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1)
    N2 = 0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1)
    N3 = 0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1)
    N4 = 0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1)
    N5 = 0.5 * (1 - xi ** 2) * (1 - eta)
    N6 = 0.5 * (1 + xi) * (1 - eta ** 2)
    N7 = 0.5 * (1 - xi ** 2) * (1 + eta)
    N8 = 0.5 * (1 - xi) * (1 - eta ** 2)
    return torch.tensor([N1, N2, N3, N4, N5, N6, N7, N8])

def dshape_dxi(xi, eta):
    # Derivatives of shape functions
    dN_dxi = torch.tensor([
        [-0.25 * (1 - eta) * (2 * xi + eta),  0.25 * (1 - eta) * (2 * xi - eta),  0.25 * (1 + eta) * (2 * xi + eta), -0.25 * (1 + eta) * (2 * xi - eta), xi * (1 - eta), 0.5 * (1 - eta**2), -xi * (1 + eta), -0.5 * (1 - eta**2)],
        [-0.25 * (1 - xi) * (xi + 2 * eta), -0.25 * (1 + xi) * (xi - 2 * eta),  0.25 * (1 + xi) * (xi + 2 * eta),  0.25 * (1 - xi) * (xi - 2 * eta), -0.5 * (1 - xi**2), -eta * (1 + xi), 0.5 * (1 - xi**2), -eta * (1 - xi)]
    ])
    return dN_dxi

def jacobian_matrix(x, dN_dxi):
    J = torch.zeros(2, 2)
    J[0, 0] = torch.dot(dN_dxi[0, :], x[:, 0])
    J[0, 1] = torch.dot(dN_dxi[0, :], x[:, 1])
    J[1, 0] = torch.dot(dN_dxi[1, :], x[:, 0])
    J[1, 1] = torch.dot(dN_dxi[1, :], x[:, 1])
    return J

def gauss_quadrature(nqp):
    if nqp == 4:
        # Coordinates and weights for a 2x2 Gauss-Legendre Quadrature
        qpt = torch.tensor([
            [-0.577350269189626, -0.577350269189626],
            [0.577350269189626, -0.577350269189626],
            [0.577350269189626, 0.577350269189626],
            [-0.577350269189626, 0.577350269189626]
        ])
        wgt = torch.tensor([1, 1, 1, 1])
    elif nqp == 9:
        # Coordinates and weights for a 3x3 Gauss-Legendre Quadrature
        qpt = torch.tensor([
            [-0.774596669241483, -0.774596669241483],
            [0, -0.774596669241483],
            [0.774596669241483, -0.774596669241483],
            [-0.774596669241483, 0],
            [0, 0],
            [0.774596669241483, 0],
            [-0.774596669241483, 0.774596669241483],
            [0, 0.774596669241483],
            [0.774596669241483, 0.774596669241483]
        ])
        wgt = torch.tensor([
            0.555555555555556, 0.888888888888889, 0.555555555555556,
            0.888888888888889, 1.33333333333333, 0.888888888888889,
            0.555555555555556, 0.888888888888889, 0.555555555555556
        ])
    else:
        raise ValueError("Unsupported number of quadrature points")
    return qpt, wgt

def create_annular_mesh(inner_radius, outer_radius, num_radial_divisions, num_circumferential_divisions):
    theta = np.linspace(0, 2 * np.pi, num_circumferential_divisions, endpoint=False)
    r = np.linspace(inner_radius, outer_radius, num_radial_divisions)
    theta, r = np.meshgrid(theta, r)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    nodes = torch.tensor(np.column_stack([x.flatten(), y.flatten()]), dtype=torch.float64)

    mid_nodes = []
    for i in range(num_radial_divisions):
        for j in range(num_circumferential_divisions):
            if i < num_radial_divisions - 1:
                mid_x_r = (x[i, j] + x[i + 1, j]) / 2
                mid_y_r = (y[i, j] + y[i + 1, j]) / 2
                mid_nodes.append([mid_x_r, mid_y_r])
            next_j = (j + 1) % num_circumferential_divisions
            mid_x_c = (x[i, j] + x[i, next_j]) / 2
            mid_y_c = (y[i, j] + y[i, next_j]) / 2
            mid_nodes.append([mid_x_c, mid_y_c])

    mid_nodes = torch.tensor(mid_nodes, dtype=torch.float64)
    nodes = torch.cat((nodes, mid_nodes), dim=0)

    elements = []
    num_nodes = num_radial_divisions * num_circumferential_divisions
    num_mid_nodes = len(mid_nodes)
    for i in range(num_radial_divisions - 1):
        for j in range(num_circumferential_divisions):
            n1 = i * num_circumferential_divisions + j
            n2 = n1 + num_circumferential_divisions
            n3 = n2 + 1 if (j != num_circumferential_divisions - 1) else n2 + 1 - num_circumferential_divisions
            n4 = n1 + 1 if (j != num_circumferential_divisions - 1) else n1 + 1 - num_circumferential_divisions
            n5 = num_nodes + i * num_circumferential_divisions + j
            n6 = num_nodes + i * num_circumferential_divisions + (j + 1) % num_circumferential_divisions
            n7 = n5 + num_circumferential_divisions
            n8 = n6 + num_circumferential_divisions
            elements.append([n1, n2, n3, n4, n5, n6, n7, n8])

    return nodes, torch.tensor(elements, dtype=torch.long)

def analysis():
    # Material properties
    E = 210e9  # Young's Modulus in Pa
    nu = 0.3   # Poisson's Ratio
    thickness = 0.05  # Thickness in meters

    # Mesh parameters
    R_inner = 0.185  # meters
    R_outer = 0.370  # meters
    n_radial = 4
    n_circum = 8

    # Create the mesh
    nodes, elements = create_annular_mesh(R_inner, R_outer, n_radial, n_circum)

    # Debugging: Print nodes and elements
    print("Nodes:")
    print(nodes)
    print("Elements:")
    print(elements)

    # Check if all element indices are within the range of nodes
    num_nodes = nodes.size(0)
    for e in elements:
        if torch.any(e >= num_nodes):
            print(f"Error: Element {e} contains indices out of bounds")

    # Gauss quadrature
    qpt, wgt = gauss_quadrature(4)

    # Setup for analysis (stiffness matrix, force vector, etc.)
    ndof = nodes.size(0) * 2  # Number of degrees of freedom
    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    print(K.shape)
    F = torch.zeros(ndof, dtype=torch.float64)

    # Define material matrix
    C = E / (1 - nu**2) * torch.tensor([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

    # Iterate over elements to assemble global stiffness matrix
    for e in elements:
        ke = torch.zeros((16, 16), dtype=torch.float64)  # Element stiffness matrix
        element_nodes = nodes[e]
        for i, pt in enumerate(qpt):
            xi, eta = pt
            N = shape_functions(xi, eta)
            dN_dxi = dshape_dxi(xi, eta)
            J = jacobian_matrix(element_nodes, dN_dxi)
            detJ = torch.det(J)
            dN_dxy = torch.matmul(torch.inverse(J), dN_dxi)

            B = torch.zeros((3, 16))
            B[0, 0::2] = dN_dxy[0, :]
            B[1, 1::2] = dN_dxy[1, :]
            B[2, 0::2] = dN_dxy[1, :]
            B[2, 1::2] = dN_dxy[0, :]

            ke += torch.matmul(torch.matmul(B.T, C), B) * detJ * wgt[i]

        dof_indices = torch.tensor([2 * node for node in e] + [2 * node + 1 for node in e], dtype=torch.long)
        for i in range(16):
            for j in range(16):
                K[dof_indices[i], dof_indices[j]] += ke[i, j]

    # Solve the system (simplified, without boundary conditions for demonstration)
    u = torch.linalg.solve(K, F)

    # Postprocessing: visualize the deformation (simplified)
    # This section needs proper displacement handling and visualization setup
    print("Displacements:", u)

if __name__ == "__main__":
    analysis()
