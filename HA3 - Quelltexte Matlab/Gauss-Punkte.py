import numpy as np
import matplotlib.pyplot as plt
import torch
import math

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

disp_scaling = 1000
toplot = True

x = torch.reshape(torch.linspace(0, 70, 11), [-1, 1])
print("x: ", x)

conn = torch.from_numpy(np.array([[1,  3,  2], [3,  5,  4], [5,  7,  6], [7,  9,  8], [9,  11,  10]]))

# Boundary conditions
drltDofs = torch.from_numpy(np.array([1]))  # Global DOF numbers where Dirichlet DOF 's are prescribed
freeDofs = torch.from_numpy(np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))  # Global DOF numbers where displacement is unknown, Free DOF's
u_d = torch.from_numpy(np.array([0.]))  # Value of the displacement at the prescribed nodes

f_sur = torch.from_numpy(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (300 + 75) * 9.81]))

# Constant body force
b = 7850 * 9.81

# Material parameters for the two elements
E = 2.1e11 * torch.ones(conn.size()[0], 1)
area = 3 * 89.9e-6 * torch.ones(conn.size()[0], 1)

#% Fuer die Ausgabe: Verschiebungen um Faktor ueberhoehen
scalingfactor = 1000

############ Preprocessing ###############
# Extract nnp, ndm and ndf from the vector 'x'
# Hint: use MATLAB function size()
nnp = x.size()[0]
print("nnp: ", nnp)
ndm = 1
ndf = 1

# Extract nel and nen from the matrix 'conn'
# Hint: use MATLAB function size()
nel = conn.size()[0]
nen = conn.size()[1]

############# Solver #############
# Initialisation of global vectors and matrix
u = torch.zeros(nnp * ndf, 1)
K = torch.zeros(nnp * ndf, nnp * ndf)
fext = torch.zeros(nnp * ndf, 1)
fvol = torch.zeros(nnp * ndf, 1)
frea = torch.zeros(nnp * ndf, 1)

def gauss1d(nqp):
    if nqp == 1:
        xi = torch.tensor([0.0])
        w8 = torch.tensor([2.0])
    elif nqp == 2:
        xi = torch.tensor([-0.5773502691896257, 0.5773502691896257])
        w8 = torch.tensor([1.0, 1.0])
    elif nqp == 3:
        xi = torch.tensor([-0.7745966692414834, 0.0, 0.7745966692414834])
        w8 = torch.tensor([0.5555555555555556, 0.8888888888888888, 0.5555555555555556])
    else:
        raise ValueError("Unsupported number of quadrature points")
    return (xi, w8)

# 修改形状函数和其导数
def shape1d(xi, nen):
    N = torch.zeros(nen, 1)
    gamma = torch.zeros(nen, 1)

    if nen == 2:
        N[0] = 0.5 * (1 - xi)
        N[1] = 0.5 * (1 + xi)
        gamma[0] = -0.5
        gamma[1] = 0.5
    elif nen == 3:
        N[0] = 0.5 * xi * (xi - 1)
        N[2] = 1 - xi**2
        N[1] = 0.5 * xi * (xi + 1)
        gamma[0] = xi - 0.5
        gamma[2] = -2 * xi
        gamma[1] = xi + 0.5
    else:
        raise ValueError("Unknown number of element nodes")
    return (N, gamma)

def jacobian1d(xe, gamma, nen):
    Jq = torch.zeros(1,1)
    for A in range(nen):
        Jq += xe[A] * gamma[A, 0]
    detJq = torch.det(Jq)
    if detJq <= 0:
        raise ValueError("Error: detJq = ", detJq, "<= 0")
    invJq = 1 / detJq
    return (detJq, invJq)


########### SOLVER  ############
################ Create stiffness matrix and fvol ############
def solve_fem(nqp):
    nnp = x.size()[0]
    ndm = 1
    ndf = 1
    nel = conn.size()[0]
    nen = conn.size()[1]

    u = torch.zeros(nnp * ndf, 1)
    K = torch.zeros(nnp * ndf, nnp * ndf)
    fext = torch.zeros(nnp * ndf, 1)
    fvol = torch.zeros(nnp * ndf, 1)
    frea = torch.zeros(nnp * ndf, 1)

    for e in range(nel):
        xe = x[conn[e, :] - 1]

        (xi, w8) = gauss1d(nqp)

        for q in range(nqp):
            (N, gamma) = shape1d(xi[q], nen)
            (detJq, invJq) = jacobian1d(xe, gamma, nen)

            G = gamma * invJq

            for A in range(nen):
                fvol[conn[e, A] - 1, 0] += w8[q] * detJq * N[A, 0] * b * area[e, 0]
                for B in range(nen):
                    K[conn[e, A] - 1, conn[e, B] - 1] += w8[q] * E[e, 0] * area[e, 0] * (G[A, 0] * G[B, 0]) * detJq

    solve_K = K.clone()
    for i in range(drltDofs.size()[0]):
        solve_K[drltDofs[i] - 1, drltDofs[i] - 1] = 1e13

    u[:, 0] = torch.linalg.solve(solve_K, fvol[:, 0] + f_sur - torch.tensordot(K, u_d, dims=[[1], [0]]))
    u[drltDofs - 1, 0] = u_d

    frea[drltDofs - 1, 0] = torch.sum(u[drltDofs - 1, :] * K[:, drltDofs - 1]) - (fext[drltDofs - 1, :] + fvol[drltDofs - 1, :])
    fext = torch.matmul(K, u)

    eps = torch.zeros(nel * nqp, 1)
    sigma = torch.zeros(nel * nqp, 1)
    x_eps = torch.zeros(nel * nqp, 1)

    for e in range(nel):
        xe = x[conn[e, :] - 1]
        (xi, w8) = gauss1d(nqp)

        for q in range(nqp):
            (N, gamma) = shape1d(xi[q], nen)
            (detJq, invJq) = jacobian1d(xe, gamma, nen)
            G = gamma * invJq

            eps[e * nqp + q] = torch.tensordot(torch.transpose(u[conn[e, :] - 1], 0, 1), G[:], dims=[[1], [0]])
            sigma[e * nqp + q] = E[e, 0] * eps[e * nqp + q]
            x_eps[e * nqp + q] = torch.tensordot(xe, N)

    return u, fext, fvol, frea, x_eps, sigma

###### Post-processing/ plots ########
plt.figure(figsize=(15, 10))

nqps = [2, 3]
colors = ['k', 'g']
labels = ['nqp=2', 'nqp=3']
sigma_list = []
u_list = []
# Displacement plot
plt.subplot(4, 1, 1)
for nqp, color, label in zip(nqps, colors, labels):
    u, _, _, _, _, _ = solve_fem(nqp)
    u_list.append(u)
    plt.plot(x, u, color + 'x-', label=label)
plt.title('Displacement')
plt.legend()

# Forces plot
plt.subplot(4, 1, 2)
for nqp, color, label in zip(nqps, colors, labels):
    _, fext, fvol, frea, _, _ = solve_fem(nqp)
    plt.plot(x, fext, color + 'o', label=label + ' fext')
    plt.plot(x, fvol, color + 'x', label=label + ' fvol')
    plt.plot(x, frea, color + 's', label=label + ' frea')
plt.title('Forces')
plt.legend()

# Stress plot
plt.subplot(4, 1, 3)
for nqp, color, label in zip(nqps, colors, labels):
    _, _, _, _, x_eps, sigma = solve_fem(nqp)
    sigma_list.append(sigma)
    plt.plot(x_eps, sigma, color + 'x-', label=label)
plt.title('Stress and X_eps')
plt.legend()

plt.tight_layout()
plt.show()

















