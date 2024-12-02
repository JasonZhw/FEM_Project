import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
disp_scaling = 3*1e4
toplot = True

# Function to generate mesh for a rectangular plate
def generate_plate_mesh(length, width, num_elements_length, num_elements_width):
    x = np.linspace(0, length, num_elements_length + 1)
    y = np.linspace(0, width, num_elements_width + 1)
    nodes = np.array([(i, j) for j in y for i in x])
    elements = []
    for j in range(num_elements_width):
        for i in range(num_elements_length):
            n1 = j * (num_elements_length + 1) + i
            n2 = n1 + 1
            n3 = n1 + num_elements_length + 2
            n4 = n1 + num_elements_length + 1
            elements.append([n1, n2, n3, n4])
    elements = np.array(elements)
    return nodes, elements

def shape_functions(xi, eta):
    N = torch.zeros(4)
    N[0] = (1 - xi) * (1 - eta) / 4
    N[1] = (1 + xi) * (1 - eta) / 4
    N[2] = (1 + xi) * (1 + eta) / 4
    N[3] = (1 - xi) * (1 + eta) / 4
    return N

def shape_function_derivatives(xi, eta):
    dN_dxi = torch.zeros(4)
    dN_deta = torch.zeros(4)
    dN_dxi[0] = -(1 - eta) / 4
    dN_dxi[1] = (1 - eta) / 4
    dN_dxi[2] = (1 + eta) / 4
    dN_dxi[3] = -(1 + eta) / 4
    dN_deta[0] = -(1 - xi) / 4
    dN_deta[1] = -(1 + xi) / 4
    dN_deta[2] = (1 + xi) / 4
    dN_deta[3] = (1 - xi) / 4
    return dN_dxi, dN_deta

def gauss_quadrature():
    gauss_points = torch.tensor([
        [-1 / math.sqrt(3), -1 / math.sqrt(3)],
        [1 / math.sqrt(3), -1 / math.sqrt(3)],
        [1 / math.sqrt(3), 1 / math.sqrt(3)],
        [-1 / math.sqrt(3), 1 / math.sqrt(3)]
    ])
    weights = torch.tensor([1, 1, 1, 1])
    return gauss_points, weights

def jacobian(x, dN_dxi, dN_deta):
    J = torch.zeros(2, 2)
    nen = x.shape[0]
    for i in range(nen):
        J[0, 0] += dN_dxi[i] * x[i, 0]
        J[0, 1] += dN_dxi[i] * x[i, 1]
        J[1, 0] += dN_deta[i] * x[i, 0]
        J[1, 1] += dN_deta[i] * x[i, 1]
    return J
# Function to calculate stress and strain from displacement
def calculate_stress(B, D, u_elem):
    strain = B @ u_elem
    stress = D @ strain
    return stress, strain

E = 210e9
nu = 0.3
ndf = 2
nen = 4
length = 1.0
width = 0.2
num_elements_length = 30
num_elements_width = 5
# Generate the mesh
x, elems = generate_plate_mesh(length, width, num_elements_length, num_elements_width)
x = torch.from_numpy(x)
elems = torch.from_numpy(elems).to(torch.long)
# Initialize Dirichlet and Neumann boundary condition arrays
num_nodes = x.shape[0]
drlt = torch.zeros(num_nodes, 2)
neum = torch.zeros(num_nodes, 2)

# Apply boundary conditions
left_nodes = torch.where(x[:, 0] == 0)[0]
for node in left_nodes:
    drlt[node, 0] = 1
    drlt[node, 1] = 1
upper_nodes = torch.where(x[:,1]==0.20)[0]
F_total = 1e4
for node in upper_nodes:
    neum[node, 0] = F_total / len(upper_nodes)
'''right_nodes = torch.where(x[:, 0] == length)[0]
F_total = 1e6
for node in right_nodes:
    neum[node, 0] = F_total / len(right_nodes)'''


plt.figure(figsize=(10, 2))
for elem in elems:
    elem = elem.numpy()
    plt.plot(x[elem[[0, 1, 2, 3, 0]], 0].numpy(), x[elem[[0, 1, 2, 3, 0]], 1].numpy(), 'blue')
plt.scatter(x[:, 0].numpy(), x[:, 1].numpy(), c='black')
plt.title('Initial Mesh')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.show()

# Initialize global stiffness matrix and force vector
gauss_points, weights = gauss_quadrature()
K = torch.zeros((num_nodes * ndf, num_nodes * ndf), dtype=torch.float64)
F = torch.zeros(num_nodes * ndf, dtype=torch.float64)
# Assemble the global force vector
for node in range(num_nodes):
    F[2 * node] += neum[node, 0]
    F[2 * node + 1] += neum[node, 1]
# Assemble the global stiffness matrix
for elem in elems:
    Ke = torch.zeros((nen * ndf, nen * ndf), dtype=torch.float64)
    xe = x[elem, :]
    for gp, w in zip(gauss_points, weights):
        xi, eta = gp
        N = shape_functions(xi, eta)
        dN_dxi, dN_deta = shape_function_derivatives(xi, eta)
        J = jacobian(xe, dN_dxi, dN_deta)
        detJ = torch.det(J)
        invJ = torch.inverse(J)
        dN_dx = torch.zeros(nen, dtype=torch.float64)
        dN_dy = torch.zeros(nen, dtype=torch.float64)
        for i in range(nen):
            dN_dx[i] = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i]
            dN_dy[i] = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i]
        B = torch.zeros((3, nen * ndf), dtype=torch.float64)
        for i in range(nen):
            B[0, 2 * i] = dN_dx[i]
            B[1, 2 * i + 1] = dN_dy[i]
            B[2, 2 * i] = dN_dy[i]
            B[2, 2 * i + 1] = dN_dx[i]
        D = (E / (1 - nu ** 2)) * torch.tensor([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]], dtype=torch.float64)
        Ke += B.t().mm(D).mm(B) * detJ * w
    for i in range(nen):
        for j in range(nen):
            K[2 * elem[i]:2 * elem[i] + 2, 2 * elem[j]:2 * elem[j] + 2] += Ke[2 * i:2 * i + 2, 2 * j:2 * j + 2]
# Apply boundary conditions in the global stiffness matrix and force vector
for node in range(num_nodes):
    if drlt[node, 0] == 1:
        idx = 2 * node
        K[idx, :] = 0
        K[:, idx] = 0
        K[idx, idx] = 1
        F[idx] = 0
    if drlt[node, 1] == 1:
        idx = 2 * node + 1
        K[idx, :] = 0
        K[:, idx] = 0
        K[idx, idx] = 1
        F[idx] = 0
# Solve the system of equations to find displacements
u = torch.linalg.solve(K, F)
print("u:", u)
# Reshape displacement vector for easier handling
u_reshaped = torch.reshape(u, (-1, 2))
x_disped = x + disp_scaling * u_reshaped

plt.figure(figsize=(10, 2))
for elem in elems:
    elem = elem.numpy()
    plt.plot(x_disped[elem[[0, 1, 2, 3, 0]], 0].numpy(), x_disped[elem[[0, 1, 2, 3, 0]], 1].numpy(), 'green')
plt.scatter(x_disped[:, 0].numpy(), x_disped[:, 1].numpy(), c='black')
plt.title('Deformed Shape')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.show()
# Initialize arrays for stress and node contributions
stress_all_nodes = [torch.zeros((3,), dtype=torch.float64) for _ in range(num_nodes)]
node_contributions = [0 for _ in range(num_nodes)]
# Calculate stress at each node
for elem in elems:
    xe = x[elem, :]
    ue = u_reshaped[elem].reshape(-1)
    for gp, w in zip(gauss_points, weights):
        xi, eta = gp
        dN_dxi, dN_deta = shape_function_derivatives(xi, eta)
        J = jacobian(xe, dN_dxi, dN_deta)
        detJ = torch.det(J)
        invJ = torch.inverse(J)
        dN_dx = torch.zeros(nen, dtype=torch.float64)
        dN_dy = torch.zeros(nen, dtype=torch.float64)
        for i in range(nen):
            dN_dx[i] = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i]
            dN_dy[i] = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i]
        B = torch.zeros((3, nen * ndf), dtype=torch.float64)
        for i in range(nen):
            B[0, 2 * i] = dN_dx[i]
            B[1, 2 * i + 1] = dN_dy[i]
            B[2, 2 * i] = dN_dy[i]
            B[2, 2 * i + 1] = dN_dx[i]
        stress, strain = calculate_stress(B, D, ue)
        for i in range(nen):
            stress_all_nodes[elem[i]] += stress / len(gauss_points)
            node_contributions[elem[i]] += 1

# Average stress values at each node
for idx in range(num_nodes):
    if node_contributions[idx] > 0:
        stress_all_nodes[idx] /= node_contributions[idx]
# Extract stress components for plotting
stress_x_all_nodes = torch.tensor([s[0] for s in stress_all_nodes])
stress_y_all_nodes = torch.tensor([s[1] for s in stress_all_nodes])
tau_xy_all_nodes = torch.tensor([s[2] for s in stress_all_nodes])
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

axs[0, 0].scatter(x[:, 0].numpy(), x[:, 1].numpy(), c=stress_x_all_nodes.numpy(), cmap='jet', s=5)
axs[0, 0].set_title('Stress Distribution in x direction')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
axs[0, 0].axis('equal')
fig.colorbar(axs[0, 0].collections[0], ax=axs[0, 0], label='Stress σ_x')

axs[0, 1].scatter(x[:, 0].numpy(), x[:, 1].numpy(), c=stress_y_all_nodes.numpy(), cmap='jet', s=5)
axs[0, 1].set_title('Stress Distribution in y direction')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
axs[0, 1].axis('equal')
fig.colorbar(axs[0, 1].collections[0], ax=axs[0, 1], label='Stress σ_y')


axs[1, 0].scatter(x[:, 0].numpy(), x[:, 1].numpy(), c=tau_xy_all_nodes.numpy(), cmap='jet', s=5)
axs[1, 0].set_title('Shear Stress τ_xy')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')
axs[1, 0].axis('equal')
fig.colorbar(axs[1, 0].collections[0], ax=axs[1, 0], label='Shear Stress τ_xy')
def von_mises_stress(stress):
    sigma_x, sigma_y, tau_xy = stress[0], stress[1], stress[2]
    return torch.sqrt(sigma_x**2 + sigma_y**2 - sigma_x * sigma_y + 3 * tau_xy**2)
von_mises_stress_all_nodes = torch.tensor([von_mises_stress(s) for s in stress_all_nodes])
axs[1, 1].scatter(x[:, 0].numpy(), x[:, 1].numpy(), c=von_mises_stress_all_nodes.numpy(), cmap='jet', s=5)
axs[1, 1].set_title('Von Mises Stress')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')
axs[1, 1].axis('equal')
fig.colorbar(axs[1, 1].collections[0], ax=axs[1, 1], label='Von Mises Stress')
