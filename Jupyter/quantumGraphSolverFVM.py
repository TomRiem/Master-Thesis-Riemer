import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class QuantumGraphSolverFVM(object):

    def __init__(self, graph):
        self.graph = graph

    def solve(self, nx, nt, eta=1):

        self.nx = nx

        # Abbreviate function f
        f = self.graph.f
        eps = self.graph.eps

        nxi = nx - 2
        n_edges = self.graph.ne
        n_nodes = self.graph.n_v

        n_inner = nxi * n_edges
        n_dofs = n_inner + n_nodes

        # Model parameters
        L = self.graph.ub[1] - self.graph.lb[1]
        self.L = L
        he = L / nx

        idxi = np.arange(nxi)
        # idxn = np.arange(self.graph.n_v)
        tau = (self.graph.ub[0] - self.graph.lb[0]) / nt

        # Number of edges adjacent to vertex
        nout = np.array([len(x) for x in self.graph.Vout])
        nin = np.array([len(x) for x in self.graph.Vin])
        n_adj = nout + nin

        def get_u_edge(u, edgeidx, timeidx=-1):
            if timeidx == -1:
                uk = u
            else:
                uk = u[:, timeidx]

            i = edgeidx

            idx_in = n_inner + self.graph.E[i][0]
            idx_inner = i * nxi + idxi
            idx_out = n_inner + self.graph.E[i][1]

            # All entries on the edge
            u_ = np.hstack((uk[idx_in:idx_in + 1],
                            uk[idx_inner],
                            uk[idx_out:idx_out + 1]))

            return u_

        self.get_u_edge = get_u_edge

        e = np.ones(nxi)
        SblkII = []
        SblkIV = []
        SblkVV = []

        Mdiag = []

        for k in range(self.graph.ne):

            # This could be outside as long as length in constant
            M_II = sp.sparse.dia_matrix(
                (e * he, np.array([0])), shape=(nxi, nxi))
            D_II = sp.sparse.dia_matrix(
                (np.array([-e, 2 * e, -e]) / he, np.array([-1, 0, 1])),
                shape=(nxi, nxi))
            ek = self.graph.E[k]
            D_IV = sp.sparse.csr_matrix((np.array(
                [-1 / he, -1 / he]),
                (np.array([0, nxi - 1]), np.array(ek))),
                shape=(nxi, self.graph.n_v))
            S_II = M_II + tau * eps * D_II
            S_IV = tau * eps * D_IV

            SblkII.append(S_II)
            SblkIV.append(S_IV)
            Mdiag.append(M_II)

        # Assemble stiffness matrix
        SblkII = sp.sparse.block_diag(SblkII)
        SblkIV = sp.sparse.vstack(SblkIV)
        M_VV = sp.sparse.dia_matrix(
            (n_adj * he, np.array([0])), shape=(n_nodes, n_nodes))
        D_VV = sp.sparse.dia_matrix(
            (n_adj / he, np.array([0])), shape=(n_nodes, n_nodes))
        SblkVV = M_VV + tau * eps * D_VV
        S = sp.sparse.bmat(
            [[SblkII, SblkIV], [SblkIV.transpose(), SblkVV]], format='csc')

        # Assemble mass matrix
        Mdiag.append(M_VV)
        M = sp.sparse.block_diag(Mdiag)

        Sop = sp.sparse.linalg.splu(S)
        u = np.zeros((n_dofs, nt))


        for k in range(1, nt):


            # Assemble right-hand side
            # Assemble advection term
            F = np.zeros((n_dofs,))

            uk = u[:, k - 1]

            # test functions in the interior of each edge
            for i in range(n_edges):

                F_ = np.zeros((nxi,))

                # All entries on the edge
                u_ = self.get_u_edge(uk, i)

                # Flux at right and left end of the interval
                F_ += (f(u_[2:]) + f(u_[1:-1])) / 2 \
                    + 0.5 * eta * (u_[1:-1] - u_[2:])
                F_ += - (f(u_[1:-1]) + f(u_[0:-2])) / 2 \
                    + 0.5 * eta * (u_[1:-1] - u_[0:-2])

                # Insert into global flux vector
                F[i * nxi + idxi] = F_

            # test functions in vertices
            for i in range(n_nodes):
                u_v = uk[n_inner + i]

                for j in self.graph.Vout[i]:

                    u_e = uk[j * nxi]
                    F[n_inner + i] += (f(u_v) + f(u_e)) / 2 \
                        + .5 * eta * (u_v - u_e)
                for j in self.graph.Vin[i]:

                    u_e = uk[(j + 1) * nxi - 1]
                    F[n_inner + i] += -(f(u_v) + f(u_e)) / 2 \
                        + .5 * eta * (u_v - u_e)

            for i in self.graph.inflowNodes:
                u_v = uk[n_inner + i]
                F[n_inner + i] += -self.graph.dirichletAlpha[i] * (1 - u_v)

            for i in self.graph.outflowNodes:
                u_v = uk[n_inner + i]
                F[n_inner + i] += self.graph.dirichletBeta[i] * u_v

            # Set up right-hand side
            rhs = M * uk - tau * F

            # Solve equation system
            u[:, k] = Sop.solve(rhs)

        self.u = u

        return u

    def plotNetwork(self, j=0, u=None, fig=None):
        X = np.linspace(0, self.L, self.nx).reshape((-1, 1))
        pos = self.graph.pos
        E = self.graph.E
        xy_list = [pos[e[0]] + X * (pos[e[1]] - pos[e[0]]) / self.L for e in E]
        if fig is None:
            fig = plt.figure(1, clear=True)
        else:
            fig.clf()

        if u is None:
            u = self.u

        ax = fig.add_subplot(1, 1, 1, projection='3d')

        for i, e in enumerate(self.graph.E):
            uk = self.get_u_edge(u, i, j)
            ax.plot(xy_list[i][:, 0], xy_list[i][:, 1], uk)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlim([.0, 1.0])
        # ax.view_init(12, 135)
        ax.view_init(12, 290)

        return ax
