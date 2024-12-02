import numpy as np
import matplotlib.pyplot as plt
import torch
import math

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

disp_scaling = 1000
toplot = True

# Define coordinates for a 2D quadrilateral mesh
x = torch.reshape(torch.linspace(0, 70, 11), [-1, 1])
print("x: ", x)
print("x: ", x)

# Define connectivity list for 2D quadrilateral elements
conn = torch.from_numpy(np.array([
    [1, 2, 3, 4]
]))

# Number of quadrature points per Element, nqp
nqp = 2

# Boundary conditions
drltDofs = torch.from_numpy(np.array([1])) # Global DOF numbers where Dirichlet DOF 's are prescribed
freeDofs = torch.from_numpy(np.array([2, 3, 4])) # Global DOF numbers where displacement is unknown, Free DOF's
u_d = torch.from_numpy(np.array([0.]) )# Value of the displacement at the prescribed nodes

f_sur = torch.from_numpy(np.array([0, 0, 0, 0, 0, 0, 0, 0]))

# Constant body force
b = 7850 * 9.81

# Material parameters for the two elements
E = 2.1e11 * torch.ones(conn.size()[0], 1)
area = 3 * 89.9e-6 * torch.ones(conn.size()[0], 1)

# Scaling factor for the output displacements
scalingfactor = 1
# 高斯积分点和权重
def gauss2d(nqp):
    if nqp == 2:
        xi = torch.tensor([[-1/np.sqrt(3), -1/np.sqrt(3)],
                           [1/np.sqrt(3), -1/np.sqrt(3)],
                           [1/np.sqrt(3), 1/np.sqrt(3)],
                           [-1/np.sqrt(3), 1/np.sqrt(3)]], dtype=torch.float64)
        w8 = torch.tensor([1, 1, 1, 1], dtype=torch.float64)
    else:
        raise ValueError("Unsupported number of quadrature points")
    return xi, w8

# 形函数和导数
def shape2d_quad(xi_eta):
    xi, eta = xi_eta
    N = torch.tensor([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta)
    ], dtype=torch.float64)
    dN_dxi = torch.tensor([
        [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)],
        [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)]
    ], dtype=torch.float64)
    return N, dN_dxi
def jacobian2d(xe, dN_dxi):
    J = dN_dxi @ xe
    detJ = torch.det(J)
    invJ = torch.inverse(J)
    return detJ, invJ
nnp = x.size(0)  # Number of nodes
ndm = 2          # Number of spatial dimensions (2D problem)
ndf = 2          # Number of degrees of freedom per node (u and v)

nel = conn.size(0)  # Number of elements
nen = conn.size(1)  # Number of element nodes

# Initialisation of global vectors and matrix
u = torch.zeros(nnp * ndf, 1)
K = torch.zeros(nnp * ndf, nnp * ndf)
fext = torch.zeros(nnp * ndf, 1)
fvol = torch.zeros(nnp * ndf, 1)
frea = torch.zeros(nnp * ndf, 1)
for e in range(nel):
    xe = x[conn[e, :] - 1]
    xi, w8 = gauss2d(nqp)
    for q in range(nqp):
        N, dN_dxi = shape2d_quad(xi[q])
        detJ, invJ = jacobian2d(xe, dN_dxi)
        dN_dx = invJ @ dN_dxi

        B = torch.zeros((3, 8), dtype=torch.float64)
        B[0, 0::2] = dN_dx[0, :]
        B[1, 1::2] = dN_dx[1, :]
        B[2, 0::2] = dN_dx[1, :]
        B[2, 1::2] = dN_dx[0, :]

        D = torch.tensor([
            [1, 0.3, 0],
            [0.3, 1, 0],
            [0, 0, (1 - 0.3) / 2]
        ], dtype=torch.float64) * (E[e, 0] / (1 - 0.3 ** 2))

        ke = B.T @ D @ B * detJ * w8[q]

        for i in range(4):
            for j in range(4):
                K[2 * (conn[e, i] - 1):2 * (conn[e, i] - 1) + 2, 2 * (conn[e, j] - 1):2 * (conn[e, j] - 1) + 2] += ke[2 * i:2 * i + 2, 2 * j:2 * j + 2]

# Boundary conditions
for i in range(drltDofs.size()[0]):
    solve_K[drltDofs[i] - 1, :] = 0
    solve_K[:, drltDofs[i] - 1] = 0
    solve_K[drltDofs[i] - 1, drltDofs[i] - 1] = 1

# Solve the linear system
u = torch.linalg.solve(solve_K, fext + fvol)
u[drltDofs - 1, 0] = u_d
frea = K @ u
fext = torch.matmul(K, u)

eps = torch.zeros(nel * nqp, 1)
sigma = torch.zeros(nel * nqp, 1)
x_eps = torch.zeros(nel * nqp, 1)

for e in range(nel):
    xe = x[conn[e, :] - 1]
    xi, w8 = gauss2d(nqp)
    for q in range(nqp):
        N, dN_dxi = shape2d_quad(xi[q])
        detJ, invJ = jacobian2d(xe, dN_dxi)
        dN_dx = invJ @ dN_dxi

        B = torch.zeros((3, 8), dtype=torch.float64)
        B[0, 0::2] = dN_dx[0, :]
        B[1, 1::2] = dN_dx[1, :]
        B[2, 0::2] = dN_dx[1, :]
        B[2, 1::2] = dN_dx[0, :]

        eps[e * nqp + q] = torch.tensordot(B, u[conn[e, :] - 1].reshape(-1), dims=1)
        sigma[e * nqp + q] = D @ eps[e * nqp + q]
        x_eps[e * nqp + q] = torch.tensordot(xe, N, dims=1)

print("u: ", u)
plt.subplot(4, 1, 1)
plt.plot(x[:,
