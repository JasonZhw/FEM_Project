import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import math
import time as timemodule

# 设置默认数据类型为 float64
torch.set_default_dtype(torch.float64)

disp_scaling = 1000
toplot = True

tdm = 2

def analysis():
    E = 79308.361e6
    nu = 0.3

    ndf = 2
    ndm = 2
    global tdm

    nen = 8
    # 刹车盘尺寸
    outer_radius = 740 / 2
    inner_radius = 370 / 2
    thickness = 50
    mass = 300000  # kg
    deceleration = 0.5  # m/s^2
    friction_coefficient = 0.3
    normal_force = 1000000  # N
    friction_force = friction_coefficient * normal_force
    inertia_force = mass * deceleration

    # 使用极坐标生成刹车盘网格节点
    num_radial = 5
    num_circumferential = 12
    r = np.linspace(inner_radius, outer_radius, num_radial)
    theta = np.linspace(0, 2 * np.pi, num_circumferential, endpoint=False)
    x = np.array([[ri * np.cos(ti), ri * np.sin(ti)] for ri in r for ti in theta])
    x = torch.tensor(x, dtype=torch.float64)

    # 生成单元连接
    elems = []
    for i in range(num_radial - 1):
        for j in range(num_circumferential):
            next_j = (j + 1) % num_circumferential
            n0 = i * num_circumferential + j
            n1 = i * num_circumferential + next_j
            n2 = (i + 1) * num_circumferential + next_j
            n3 = (i + 1) * num_circumferential + j
            elems.append([n0, n1, n2, n3])
    elems = torch.tensor(elems, dtype=torch.int64)

    # 定义边界条件和力
    drlt = torch.tensor([[i, 0, 0] for i in range(num_circumferential)] +
                        [[i, 1, 0] for i in range(num_circumferential)], dtype=torch.float64)
    neum = torch.tensor([[i, 0, friction_force / num_circumferential] for i in range(num_circumferential)] +
                        [[i, 1, inertia_force / num_circumferential] for i in range(num_circumferential)], dtype=torch.float64)

    nqp = 4

    ei = torch.eye(ndm, ndm, dtype=torch.float64)
    I = torch.eye(3, 3, dtype=torch.float64)

    ndof = ndf * x.shape[0]
    nel = elems.shape[0]

    K = torch.zeros(ndof, ndof, dtype=torch.float64)
    F = torch.zeros(ndof, dtype=torch.float64)
    u = torch.zeros(ndof, dtype=torch.float64)

    qpt = torch.tensor([[-math.sqrt(3.0) / 3.0, -math.sqrt(3.0) / 3.0],
                        [math.sqrt(3.0) / 3.0, -math.sqrt(3.0) / 3.0],
                        [math.sqrt(3.0) / 3.0, math.sqrt(3.0) / 3.0],
                        [-math.sqrt(3.0) / 3.0, math.sqrt(3.0) / 3.0]], dtype=torch.float64)
    w8 = torch.tensor([1, 1, 1, 1], dtype=torch.float64)

    masterelement = [{'N': torch.zeros(nen, dtype=torch.float64), 'gamma': torch.zeros(nen, ndm, dtype=torch.float64)} for _ in range(nqp)]
    for q in range(nqp):
        xi = qpt[q, :]
        masterelement[q]['N'][0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1])
        masterelement[q]['N'][1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1])
        masterelement[q]['N'][2] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1])
        masterelement[q]['N'][3] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1])

        masterelement[q]['gamma'][0, :] = torch.tensor([-0.25 * (1.0 - xi[1]), -0.25 * (1.0 - xi[0])], dtype=torch.float64)
        masterelement[q]['gamma'][1, :] = torch.tensor([0.25 * (1.0 - xi[1]), -0.25 * (1.0 + xi[0])], dtype=torch.float64)
        masterelement[q]['gamma'][2, :] = torch.tensor([0.25 * (1.0 + xi[1]), 0.25 * (1.0 + xi[0])], dtype=torch.float64)
        masterelement[q]['gamma'][3, :] = torch.tensor([-0.25 * (1.0 + xi[1]), 0.25 * (1.0 - xi[0])], dtype=torch.float64)

    for e in range(nel):
        ke = torch.zeros(ndf * nen, ndf * nen, dtype=torch.float64)
        fe = torch.zeros(ndf * nen, dtype=torch.float64)
        xe = torch.zeros(ndm, nen, dtype=torch.float64)
        for idm in range(ndm):
            xe[idm, :] = x[elems[e, :], idm]

        for q in range(nqp):
            N = masterelement[q]['N']
            gamma = masterelement[q]['gamma']

            Je = gamma.mm(xe.t())
            invJe = torch.inverse(Je)
            detJe = torch.det(Je)

            B = torch.zeros(3, ndf * nen, dtype=torch.float64)
            for ien in range(nen):
                dN = invJe.mm(gamma[ien, :].t())
                B[:, ndf * ien: ndf * (ien + 1)] = torch.tensor([[dN[0], 0], [0, dN[1]], [dN[1], dN[0]]], dtype=torch.float64)

            C = torch.tensor([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu) / 2]
            ], dtype=torch.float64) * E / (1 - nu ** 2)

            ke += B.t().mm(C).mm(B) * detJe * w8[q]

        edof = torch.zeros(nel, ndm, nen, dtype=torch.int64)
        for idf in range(ndf):
            for ien in range(nen):
                edof[e, idf, ien] = ndf * elems[e, ien] + idf
        gdof = edof[e, :, :].t().reshape(ndf * nen)
        K[gdof, :][:, gdof] += ke

    # Apply Dirichlet boundary conditions
    for d in drlt:
        node = int(d[0])
        dof = int(d[1])
        value = d[2]
        index = ndf * node + dof
        K[index, :] = 0
        K[index, index] = 1
        F[index] = value

    # Apply Neumann boundary conditions
    for n in neum:
        node = int(n[0])
        dof = int(n[1])
        force = n[2]
        index = ndf * node + dof
        F[index] += force

    # Solve for displacements
    u = torch.linalg.solve(K, F)

    ###### Post-processing/ plots ########
    u_reshaped = torch.reshape(u, (-1, 2))
    x_disped = x + disp_scaling * u_reshaped

    voigt = torch.tensor([[0,0], [1,1], [2,2], [0, 1], [0,2], [1,2]], dtype=torch.float64)
    ei = torch.eye(3,3, dtype=torch.float64)
    plotdata = torch.zeros(7, nel*nqp, 3, dtype=torch.float64)
    for i in range(7):
        plt.subplot(3,3,i+1)
        for e in range(nel):
            els = torch.index_select(elems[e,:], 0, torch.arange(nen))
            plt.plot(x_disped[els, 0], x_disped[els, 1], 'bo-')

            xe = torch.zeros(ndm, nen, dtype=torch.float64)
            for idm in range(ndm):
                xe[idm, :] = x[elems[e, :], idm]
            ue = torch.squeeze(u[edof[e, 0:ndm, :]])

            for q in range(nqp):
                xgauss = torch.mv(torch.transpose(x_disped[elems[e, :]], 0, 1), masterelement[q]['N'])
                plotdata[i, e*nqp + q, 0:2] = xgauss

                gamma = masterelement[q]['gamma']
                Je = gamma.mm(xe.t())
                invJe = torch.inverse(Je)
                G = gamma.mm(invJe)
                h = torch.zeros(3,3, dtype=torch.float64)
                h[0:ndm, 0:ndm] = ue.mm(G)
                eps = 0.5 * (h + h.t())
                mu = E / (2 * (1 + nu))
                lame = E * nu / ((1 + nu) * (1 - 2 * nu))
                stre = 2 * mu * eps + lame * torch.trace(eps) * ei
                sigma_v = torch.sqrt(stre[0,0]**2 + stre[1,1]**2 + stre[2,2]**2 - stre[0,0]*stre[1,1] - stre[1,1]*stre[2,2] - stre[2,2]*stre[0,0] + 3 * (stre[0,1]**2 + stre[0,2]**2 + stre[1,2]**2))
                if i < 6:
                    stre_val = stre[voigt[i, 0], voigt[i, 1]]
                else:
                    stre_val = sigma_v
                plotdata[i, e * nqp + q, 2] = stre_val

        max_stre_val = torch.max(torch.abs(plotdata[i,:, 2]))+1e-12
        plt.scatter(plotdata[i,:, 0], plotdata[i,:, 1], c=plotdata[i,:, 2]/max_stre_val, s=200, cmap=mpl.cm.jet)

if __name__ == '__main__':
    start_perfcount = timemodule.perf_counter()
    analysis()
    end_perfcount = timemodule.perf_counter()
    print("Elapsed (after compilation) = {}s".format((end_perfcount - start_perfcount)))

    if toplot:
        plt.show()
