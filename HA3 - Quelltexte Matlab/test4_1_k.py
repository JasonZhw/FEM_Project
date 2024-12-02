# ========================================================================
# Numerische Implementierung der linearen FEM
#% TU Berlin
# Institute für Mechanik
# ========================================================================
# Finite Element Code for 1d elements
# ========================================================================
import numpy as np
import matplotlib.pyplot as plt
import torch
import math

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

disp_scaling = 1000
toplot = True

############ Input ###############

# Define coordinates as a column vector
# Hint: row entries - global node number
#       column entries - corresponding x coordinates
# x = [x1 ; x2 ; x3 ; x4 ; x5]
#
x = torch.reshape(torch.linspace(0, 70, 22), [-1, 1])
print("x: ", x)

# Define connectivity list as a matrix
# between local and global node numbers
# Hint: No. of rows in the 'conn' matrix = No. of elements
#       No. of columns in the 'conn' matrix = Element node numbering
# conn  | Global list
# --------------------
# e = 1 | 1  3  2
# e = 2 | 3  5  4
#conn = torch.from_numpy(np.array([[1,2],[2,3],[3,4],[4,5], [5, 6],[6,7],[7,8],[8,9], [9,10],[10,11]]))
conn = torch.from_numpy(np.array([[1,2],[2,3],[3,4],[4,5], [5, 6],[6,7],[7,8],[8,9], [9,10],[10,11],[11,12],[12,13],[13,14],[14,15], [15, 16],[16,17],[17,18],[18,19], [19,20],[20,21],[21,22]]))
#conn = torch.from_numpy(np.array([[1,2],[2,3],[3,4],[4,5]]))
# Number of quadrature points per Element, nqp
nqp = 3

# Boundary conditions
drltDofs = torch.from_numpy(np.array([1]))  # Global DOF numbers where Dirichlet DOF 's are prescribed
#freeDofs = torch.from_numpy(np.array([2, 3, 4, 5]))
#freeDofs = torch.from_numpy(np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
freeDofs = torch.from_numpy(np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12, 13, 14, 15, 16, 17, 18, 19, 20, 21,22]))  # Global DOF numbers where displacement is unknown, Free DOF's
u_d = torch.from_numpy(np.array([0.]))  # Value of the displacement at the prescribed nodes
#f_sur = torch.from_numpy(np.array([0, 0, 0, 0,0,0,0,0,0,0,(300 + 75) * 9.81]))
f_sur = torch.from_numpy(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, (60.27* 9.81)]))#300 + 75) ]))

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

"""
% Parameter fuer die Gauss-Quadratur
%% xi sind die Stellen, an denen die Funktion ausgewertet wird
%% w8 die "weights" -- zugehörige Gewichte in der Summenformel
%% Eingabe: nqp, die Anzahl der Gauss-Quadratur-Punkte (nicht Element-Knoten!!)
"""
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

"""
% Auswertung der Formfunktionen (Lagrange-Polynome) [N] und ihrer Ableitungen [gamma]
%% an der (elementlokalen) Positionen [xi] bei [nen] Knoten im Element
%% für das Masterelement
"""
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
    #elif nen == 3:
    #    N[0] = 0.5 * xi * (xi - 1)
    #    N[1] = 1 - xi**2
    #    N[2] = 0.5 * xi * (xi + 1)
    #    gamma[0] = xi - 0.5
    #    gamma[1] = -2 * xi
    #    gamma[2] = xi + 0.5
    else:
        raise ValueError("Unknown number of element nodes")
    return (N, gamma)



"""
% Abbildung zum und vom Masterelement:
% Entspricht der "Jacobian"-Matrix
% xe: Globale Koordinaten der Knoten eines Elementes
% gamma: Ableitungen der Formfunktionen (Shape Functions) am Masterelement
% nen: Number of Element Nodes
% Ausgabe: Determinante der Jacobian-Matrix und Inverse der Jacobian
"""
# 更新雅各比矩阵计算
'''
def jacobian1d(xe, gamma, nen):
    xe_sorted, _ = torch.sort(xe.squeeze())
    J = torch.tensordot(xe_sorted, gamma.squeeze(), dims=1)
    detJq = J.item()

    if detJq <= 0:
        raise ValueError("Error: detJq = ", detJq, "<= 0")

    invJq = 1 / detJq

    return (detJq, invJq)
'''
#def jacobian1d(xe, gamma, nen):
    # 打印 xe 和 gamma 的形状及其值以调试
    #print("Shape of xe:", xe.shape)
    #print("Shape of gamma:", gamma.shape)
    #print("Values of xe:", xe)
    #print("Values of gamma:", gamma)

    # 计算 Jacobian 行列式
    #Jq = sum(xe[A] * gamma[A] for A in range(nen))

    # 打印 Jq 值以调试
    #print("Value of Jq:", Jq)

    #if Jq <= 0:
        #raise ValueError(f"无效的 Jacobian 行列式: {Jq}")

    # 计算 Jacobian 的逆
    #invJq = 1 / Jq

   # return Jq, invJq
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

########### SOLVER  ############

################ Create stiffness matrix and fvol ############
for e in range(nel):
    # 提取单元节点的坐标
    #xe = x[conn[e, :] - 1].squeeze().sort().values  # 确保 xe 是一维张量，并排序节点
    xe = x[conn[e, :] - 1]
    # 打印 xe 的形状及其值以调试
    #print("Shape of xe:", xe.shape)
    #print("Values of xe:", xe)

    xi, w8 = gauss1d(nqp)
    for q in range(nqp):
        N, gamma = shape1d(xi[q], nen)

        # 打印 gamma 的形状以调试
        #print("Shape of gamma:", gamma.shape)

        detJq, invJq = jacobian1d(xe, gamma, nen)
        # 将张量转换为标量值
        #b_scalar = b.item() if isinstance(b, torch.Tensor) else b
        #Jq_scalar = Jq.item() if isinstance(Jq, torch.Tensor) else Jq
        #w8_scalar = w8[q].item() if isinstance(w8[q], torch.Tensor) else w8[q]
        #area_scalar = area[e, 0].item() if isinstance(area[e, 0], torch.Tensor) else area[e, 0]

        G = gamma * invJq

        # 更新全局体力向量和刚度矩阵
        for A in range(nen):
            #N_scalar = N[A].item() if isinstance(N[A], torch.Tensor) else N[A]
            # 体力贡献
            #fvol[conn[e, A] - 1, 0] += b_scalar * area_scalar * N_scalar * Jq_scalar * w8_scalar
            #fvol[conn[e, A] - 1, 0] += b_scalar  * N_scalar * Jq_scalar * w8_scalar
            #GA_scalar=G[A].item() if isinstance(G[A], torch.Tensor) else G[A]
            fvol[conn[e, A] - 1, 0] += w8[q] * detJq * N[A, 0] * b * area[e, 0]


            # 循环遍历节点B
            for B in range(nen):
                #GB_scalar = G[B].item() if isinstance(G[B], torch.Tensor) else G[B]
                #K[conn[e, A] - 1, conn[e, B] - 1] += E[e, 0] * area_scalar * GA_scalar * GB_scalar * Jq_scalar * w8_scalar
                K[conn[e, A] - 1, conn[e, B] - 1] += w8[q] * E[e, 0] * area[e, 0] * (G[A, 0] * G[B, 0]) * detJq


'''
    # Loop over the gauss points, Summation over all gauss points 'q'
    for q in range(nqp):
        # % Call the shape functions and its derivatives
        # Hint: input parameters - xi(q), nen
        #       output parameters - N, gamma
        #       function name - shape1d
        (N, gamma) = shape1d(xi[q], nen)

        # % Determinant of Jacobian and the inverse of Jacobian
        # at the quadrature points q
        # Hint: For this 1d-case the Jacobian is a scalar of 1x1
        #       and its inverse also a scalar
        (detJq, invJq) = jacobian1d(xe, gamma, nen)


        #Gradient of the shape functions wrt. to x
        G = gamma * invJq

        #Loop over the number of nodes A
        for A in range(nen):
            # Volume force contribution
            fvol[conn[e, A] - 1, 0] += w8[q].item() * detJq * N[A].item() * b

            # Loop over the number of nodes A
            for B in range(nen):
                K[conn[e, A] - 1, conn[e, B] - 1] += w8[q].item() * E[e].item() * area[e].item() * (G[A].item() * G[B].item()) * detJq
'''
##### Include BC, solve the linear system K u = f ######
# Calculate nodal displacements => solve linear system
# Use "array slicing" or penalty method
solve_K = K.clone()
#print("solve_K:  ", solve_K)
for i in range(drltDofs.size()[0]):
    #solve_K[drltDofs[i] - 1, :] = 0
    #solve_K[:, drltDofs[i] - 1] = 0
    solve_K[drltDofs[i] - 1, drltDofs[i] - 1] = 1e13

#u[:, 0] = torch.linalg.solve(solve_K, fvol + f_sur)[:, 0]
u[:, 0] = torch.linalg.solve(solve_K, fvol[:, 0] + f_sur - torch.tensordot(K, u_d, dims=[[1], [0]]) )
u[drltDofs - 1, 0] = u_d

###### Process results: Calculate output quantities  #######
# Compute the reaction forces at the Dirichlet boundary
print("Shape of K:", K.shape)
print("Shape of u:", u.shape)
#frea[drltDofs - 1, 0] = torch.sum(u[drltDofs - 1, :] * K[:, drltDofs - 1])
frea[drltDofs - 1, 0] = torch.sum(u[drltDofs - 1, :] * K[:, drltDofs - 1])- (fext[drltDofs - 1, :] + fvol[drltDofs - 1, :])
#fext = f_sur + frea
#Compute the total force vector
fext = torch.matmul(K, u)

# Calculate strains and stresses
eps = torch.zeros(nel * nqp, 1)
sigma = torch.zeros(nel * nqp, 1)
x_eps = torch.zeros(nel * nqp, 1)

# for all elements
for e in range(nel):
    # Coordinates of the element nodes
    # Hint: to be extracted from the global coordinate list 'x'
    #       considering the 'conn' matrix
    xe = x[conn[e] - 1]
    (xi, w8) = gauss1d(nqp)

    # for all Gauss points
    for q in range(nqp):
        # evaluate shape functions (as above)
        # evaluate jacobian
        (N, gamma) = shape1d(xi[q], nen)
        (detJq, invJq) = jacobian1d(xe, gamma, nen)
        # Gradient of the shape functions wrt. to x
        G = gamma * invJq

        # Use derivatives of shape functions, G, to calculate eps as spatial derivative of u
        #eps[e * nqp + q] = torch.tensordot(u[conn[e] - 1].T, G)
        eps[e * nqp + q] = torch.tensordot(torch.transpose(u[conn[e,:] - 1],0,1), G[:], dims=[1][0])
        # Calculate stresses from strains and Young's modulus (i.e. apply Hooke's law)
        #sigma[e * nqp + q] = E[e] * eps[e * nqp + q]
        sigma[e * nqp + q] = E[e, 0] * eps[e * nqp + q]
        # create x-axis vector to plot eps, sigma in the Gauss points (not at the nodes!)
        x_eps[e * nqp + q] = torch.tensordot(xe, N)

print("u: ", u)



###### Post-processing/ plots ########
plt.subplot(4, 1, 1)
plt.plot(x, torch.zeros_like(x), 'ko-')
plt.plot(x[drltDofs - 1], -0.02 * torch.ones_like(drltDofs), 'g^')
plt.plot(x + scalingfactor * u, torch.zeros_like(x), 'o-')
plt.plot(0, 1, 'k-')

plt.subplot(4, 1, 2)
plt.plot(x, u, 'x-')
plt.legend(["u"])

plt.subplot(4, 1, 3)
plt.plot(x, fext, 'ro')
plt.plot(x, fvol, 'cx')
plt.plot(x, frea, 'bx')
plt.legend(["$f_{ext}$", "$f_{vol}$", "$f_{rea}$"])

plt.subplot(4, 1, 4)
plt.plot(x_eps, sigma, 'x-')
plt.tight_layout()
plt.show()
