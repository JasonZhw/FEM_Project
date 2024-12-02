import numpy as np
import matplotlib.pyplot as plt
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

disp_scaling = 1000
toplot = True

# Define coordinates as a column vector
x = torch.reshape(torch.linspace(0, 70, 11), [-1, 1])

# Define connectivity list as a matrix
conn = torch.from_numpy(np.array([[1, 3, 2], [3, 5, 4], [5, 7, 6], [7, 9, 8], [9, 11, 10]]))

# Boundary conditions
drltDofs = torch.from_numpy(np.array([1]))
freeDofs = torch.from_numpy(np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
u_d = torch.from_numpy(np.array([0.]))

# Constant body force
b = 7850 * 9.81

# Material parameters for the two elements
E = 2.1e11 * torch.ones(conn.size()[0], 1)
area = 3 * 89.9e-6 * torch.ones(conn.size()[0], 1)

# Scaling factor for displacements
scalingfactor = 1000
nqp = 2
# Gauss quadrature function
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

# Shape functions and their derivatives
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

# Jacobian matrix and its inverse
def jacobian1d(xe, gamma, nen):
    #xe_sorted, _ = torch.sort(xe.squeeze())
    #J = torch.tensordot(xe, gamma[:, 0].squeeze(), dims=0)
    Jq = torch.zeros(1,1)
    for A in range(nen):
        Jq += xe[A] * gamma[A, 0]
    detJq = torch.det(Jq)

    if detJq <= 0:
        raise ValueError("Error: detJq = ", detJq, "<= 0")

    invJq = 1 / detJq

    return (detJq, invJq)

# Solver function
def solve_fem(f_sur):
    nnp = x.size()[0]
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

    try:
        u[:, 0] = torch.linalg.solve(solve_K, fvol[:, 0] + f_sur - torch.tensordot(K, u_d, dims=[[1], [0]]))
    except torch.linalg.LinAlgError:
        solve_K_pinv = torch.linalg.pinv(solve_K)
        u[:, 0] = torch.matmul(solve_K_pinv, fvol[:, 0] + f_sur - torch.tensordot(K, u_d, dims=[[1], [0]]))
    
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

# 外力条件1
f_sur1 = torch.from_numpy(np.array([0, 0, 0, 0, 0,0, 0, 0, 0, 0, 300 * 9.81]))
f_sur2 = torch.from_numpy(np.array([0, 0, 0, 0, 0,0, 0, 0, 0, 0, 375 * 9.81]))
# 外力条件2
f_sur3 = torch.from_numpy(np.array([0, 0, 0, 0, 0,86.1*0.7* 9.81, 0, 0, 0, 0, 0]))

# 求解两种外力条件下的应力
u1, _, _, _, x_eps1, sigma1 = solve_fem(f_sur1)
u2, _, _, _, x_eps2, sigma2 = solve_fem(f_sur2)
u3, _, _, _, x_eps3, sigma3 = solve_fem(f_sur3)

# 计算误差函数

# Plotting results
plt.figure(figsize=(15, 8))

# Stress plot for two different external forces
plt.subplot(2,1,1)
plt.plot(x_eps1, sigma1, 'b-', label='durch die Aufzugskabine')
plt.plot(x_eps2, sigma2, 'r-', label='durch eine einzelne Person')
plt.plot(x_eps3, sigma3, 'k-', label='durch Eigengewicht')
plt.title('Stress for Different External Forces')
plt.legend()

plt.subplot(2,1,2)
plt.plot(x, u1, 'b-', label='durch die Aufzugskabine')
plt.plot(x, u2, 'r-', label='durch eine einzelne Person')
plt.plot(x, u3, 'k-', label='durch Eigengewicht')
plt.title('Displacement for Different External Forces')
plt.legend()

