import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import math
import time as timemodule

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

disp_scaling = 1000
toplot = True

tdm = 2


def analysis():
    E = 79308.361e6
    nu = 0.3

    ndf = 2
    ndm = 2
    global tdm

    nen = 4
    # 1-based indexing to resemble other FE input formats
    x = torch.from_numpy(np.array(
        [[0., 0.], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]]))  # [[0,0], [1,0], [1, 1], [0,1]]))
    # 1 Element with nodes 1, 2, 3, 4
    elems = torch.from_numpy(np.array([[1, 2, 5, 4], [2, 3, 6, 5], [4, 5, 8, 7], [5, 6, 9, 8]]))  ##[[1, 2, 3, 4]]))
    elems = elems - 1  # no converting to 0-based for inner indexing
    # Node 1, x direction, 0 movement
    # Node 1, y direction, 0 movement
    # Node 4, x direction, 0 movement
    drlt = torch.from_numpy(np.array([[1, 0, 0], [1, 1, 0], [4, 0, 0], [4, 1, 0], [7, 0, 0],
                                      [7, 1, 0]]))  ##[[1, 0, 0], [1, 1, 0], [4, 0, 0]])) #, [4, 1, 0]]))
    # Node 2 and 3, x direction, scale 0.5 force
    neum = torch.from_numpy(np.array([[3, 0, 5000000], [6, 0, 10000000], [9, 0, 5000000]]))  ##[[2, 0, 0.5], [3, 0, 0.5]]))

    # Gauss quadrature
    nqp = 4

    ############ Identity tensors ###########
    ei = torch.eye(ndm, ndm)

    I = torch.eye(3, 3)

    # I4 = torch.zeros(tdm, tdm, tdm, tdm)
    # for i in range(tdm):
    #     for j in range(tdm):
    #         for k in range(tdm):
    #             for l in range(tdm):
    #                 I4[i][j][k][l] = I[i][j] * I[k][l]
    #
    # I4sym = torch.zeros(tdm, tdm, tdm, tdm)
    # for i in range(tdm):
    #     for j in range(tdm):
    #         for k in range(tdm):
    #             for l in range(tdm):
    #                 I4sym[i][j][k][l] = 0.5 * (I[i][k] * I[j][l] + I[i][l] * I[j][k])
    #
    # I4dev = torch.zeros(tdm, tdm, tdm, tdm)
    # for i in range(tdm):
    #     for j in range(tdm):
    #         for k in range(tdm):
    #             for l in range(tdm):
    #                 I4dev[i][j][k][l] = I4sym[i][j][k][l] - 1 / 3 * I4[i][j][k][l]

    blk = E / (3 * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    #C4 = blk * I4 + 2 * mu * I4dev
    lame = blk - 2 / 3 * mu

    C4 = torch.zeros(2,2,2,2)
    C4[0, 0, 0, 0] = 1
    C4[1, 1, 1, 1] = 1
    C4[1, 1, 0, 0] = nu
    C4[0, 0, 1, 1] = nu

    C4[0, 1, 1, 0] = (1-nu)/2
    C4[1, 0, 1, 0] = (1 - nu) / 2
    C4[0, 1, 0, 1] = (1 - nu) / 2
    C4[1, 0, 0, 1] = (1 - nu) / 2

    C4 *= E/ (1 - nu ** 2)
    ############ Preprocessing ###############
    nnp = x.size()[0]
    print("nnp: ", nnp)
    nel = elems.size()[0]

    drlt_mask = torch.zeros(nnp * ndf, 1)

    drlt_vals = torch.zeros(nnp * ndf, 1)
    for i in range(drlt.size()[0]):
        drlt_mask[(int(drlt[i, 0]) - 1) * ndf + int(drlt[i, 1]), 0] = 1
        drlt_vals[(int(drlt[i, 0]) - 1) * ndf + int(drlt[i, 1]), 0] = drlt[i, 2]
    free_mask = torch.ones(nnp * ndf, 1) - drlt_mask
    drltDofs = torch.nonzero(drlt_mask)
    print("drltDofs", drltDofs)

    drlt_matrix = 1e22 * torch.diag(drlt_mask[:, 0], 0)

    neum_vals = torch.zeros(nnp * ndf, 1)
    for i in range(neum.size()[0]):
        neum_vals[(int(neum[i, 0]) - 1) * ndf + int(neum[i, 1]), 0] = neum[i, 2]

    #TODO


    indices = torch.tensor([0, 1, 2, 3, 0])
    for i in range(1, 8):
        plt.subplot(3,3,i)
        for e in range(nel):
            els = torch.index_select(elems[e,:], 0, indices)
            plt.plot( x[els, 0], x[els, 1], 'k-' )
            for q in range(nqp):
                xgauss = torch.mv ( torch.transpose(x[elems[e, :]], 0, 1),  masterelem_N[q] )
                plt.plot(xgauss[0], xgauss[1], 'ko', markersize=10)

        plt.plot(x[:, 0], x[:, 1], 'ko')

        for i in range(len(drlt)):
            if drlt[i, 1] == 0:
                plt.plot(x[drlt[i, 0]-1, 0]-0.02, x[drlt[i, 0]-1, 1], 'g>', markersize=10)
            if drlt[i, 1] == 1:
                plt.plot(x[drlt[i, 0]-1, 0], x[drlt[i, 0]-1, 1]-0.02, 'g^', markersize=10)
        for i in range(len(neum)):
            if neum[i, 1] == 0:
                plt.plot(x[neum[i, 0]-1, 0]+0.02, x[neum[i, 0]-1, 1], 'r>', markersize=10)
            if neum[i, 1] == 1:
                plt.plot(x[neum[i, 0]-1, 0], x[neum[i, 0]-1, 1]+0.02, 'r^', markersize=10)


    ############## Analysis ###############
    edof = torch.zeros(nel, ndf, nen, dtype=int)
    gdof = torch.zeros(nel, ndf * nen, dtype=int)
    for el in range(nel):
        for ien in range(nen):
            for idf in range(ndf):
                edof[el, idf, ien] = ndf * elems[el, ien] + idf
        gdof[el, :] = edof[el, :, :].t().reshape(ndf * nen)


    #TODO

    ###### Post-processing/ plots ########
    u_reshaped = torch.reshape(u, (-1, 2))


    x_disped = x + disp_scaling * u_reshaped

    voigt = torch.tensor([[0,0], [1,1], [2,2], [0, 1], [0,2], [1,2]])
    ei = torch.eye(3,3)
    plotdata = torch.zeros(7, nel*nqp, 3)
    for i in range(7):
        plt.subplot(3,3,i+1)
        for e in range(nel):
            els = torch.index_select(elems[e,:], 0, indices)
            plt.plot( x_disped[els, 0], x_disped[els, 1], 'bo-' )

            xe = torch.zeros(ndm, nen)
            for idm in range(ndm):
                xe[idm, :] = x[elems[e, :], idm]
            ue = torch.squeeze(u[edof[e, 0:ndm, :]])

            for q in range(nqp):
                xgauss = torch.mv ( torch.transpose(x_disped[elems[e, :]], 0, 1),  masterelem_N[q] )
                plotdata[i, e*nqp + q, 0:2] = xgauss

                gamma = masterelem_gamma[q]
                ##TODO
                G = gamma.mm(invJe)
                h = torch.zeros(3,3)
                h[0:ndm, 0:ndm] = ue.mm(G)
                eps = 0.5 * (h + h.t())
                stre = 2 * mu * eps + lame * torch.trace(eps) * ei
                sigma_v = 2 #TODO
                if i < 6:
                    stre_val = stre[voigt[i, 0], voigt[i, 1]]
                else:
                    stre_val = sigma_v
                plotdata[i, e * nqp + q, 2] = stre_val

        max_stre_val = torch.max(torch.abs(plotdata[i,:, 2]))+1e-12
        plt.scatter(plotdata[i,:, 0], plotdata[i,:, 1], c=plotdata[i,:, 2]/max_stre_val, s=200, cmap=mpl.cm.jet) #, xgauss[1],  c=stre_val)


if __name__ == '__main__':

    start_perfcount = timemodule.perf_counter()
    analysis()
    end_perfcount = timemodule.perf_counter()
    print("Elapsed (after compilation) = {}s".format((end_perfcount - start_perfcount)))

    if toplot:
        plt.show()