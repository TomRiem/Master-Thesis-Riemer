import networkx as netx
import numpy as np


class Graph(object):

    def __init__(self):

        self.initialized = False

    def buildGraph(self):

        # Define networkx multigraph
        self.G = netx.MultiDiGraph(self.A)

        # Get number of vertices
        self.n_v = self.A.shape[0]

        # Determine lists of edges and lengths
        self._determineEdgeList()

        # Determine list of vertices and incoming as well as outgoing edges
        self._determineVertexList()

        # Determine graph layout if necessary
        if self.pos is None:
            self.pos = netx.kamada_kawai_layout(self.G)
        else:
            if isinstance(self.pos, np.ndarray):
                self.pos = self.pos_array_to_dict(self.pos)
            else:
                raise ValueError('Check pos argument.')

        self.initialized = True

    def _determineEdgeList(self):
        """Determine edge matrix and weight vector.
        This could also be accomplished by a loop over `G.edges`:
            for e in G.edges:
                print(e)
        """

        self.E = []
        self.W = []

        for i in range(self.n_v):
            for j in range(i + 1, self.n_v):
                aij = self.A[i, j]
                if aij > 0:
                    self.E.append([i, j])
                    self.W.append(aij)

        # Get number of edges
        self.ne = len(self.E)

    def _determineVertexList(self):

        self.Vin = [[] for _ in range(self.n_v)]
        self.Vout = [[] for _ in range(self.n_v)]

        self.inflowNodes = []
        self.outflowNodes = []

        for i, e in enumerate(self.E):
            # Unpack edge
            vin, vout = e
            self.Vin[vout].append(i)
            self.Vout[vin].append(i)

        for i in range(self.n_v):
            if self.Vin[i] and (not self.Vout[i]):
                self.outflowNodes.append(i)

            if (not self.Vin[i]) and self.Vout[i]:
                self.inflowNodes.append(i)
        self.outflowNodes = np.array(self.outflowNodes)
        self.inflowNodes = np.array(self.inflowNodes)
        self.innerVertices = np.setdiff1d(
            np.arange(self.n_v), self.dirichletNodes)

    def plotGraph(self, **kwargs):

        netx.draw(self.G, pos=self.pos, with_labels=True, **kwargs)

    def pos_array_to_dict(self, pos):
        pos_dict = dict()
        for i in range(pos.shape[0]):
            pos_dict[i] = pos[i, :]
        return pos_dict


class Example0(Graph):
    def __init__(self, eps=1e-2):

        self.id = 0
        self.A = np.array([[0, 1],
                           [0, 0]], dtype=np.int16)

        self.pos = np.array([[0, 0], [1, 0]])

        # Set boundaries
        tmin = 0.
        tmax = 10.
        xmin = 0.
        xmax = 1.

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        self.dirichletNodes = np.array([0, 1])
        self.dirichletAlpha = np.array([0.7, 0.0])
        self.dirichletBeta = np.array([0.0, 0.8])

        self.eps = eps

    def f(self, u):
        return u * (1 - u)

    def df(self, u):
        return 1 - 2 * u

    def pde(self, u, ut, ux, uxx):
        return ut - self.eps * uxx + self.df(u) * ux

    def flux(self, u, ux):
        return - self.eps * ux + self.f(u)

    def initial_cond(self, x):
        return np.zeros_like(x)


class Example1(Graph):
    def __init__(self, eps=1e-2):

        self.id = 1
        self.A = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0]
                           ], dtype=np.int16)

        self.pos = None

        # Set boundaries
        tmin = 0.
        tmax = 1.
        xmin = 0.
        xmax = 0.1

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        self.dirichletNodes = np.array([0, 7])
        self.dirichletAlpha = np.zeros(8)
        self.dirichletBeta = np.zeros(8)
        self.dirichletAlpha[0] = .8
        self.dirichletBeta[7] = .5

        self.eps = eps

    def f(self, u):
        return u * (1 - u)

    def df(self, u):
        return 1 - 2 * u

    def pde(self, u, ut, ux, uxx):
        return ut - self.eps * uxx + self.df(u) * ux

    def flux(self, u, ux):
        return - self.eps * ux + self.f(u)

    def initial_cond(self, x):
        return np.zeros_like(x)


class Example2(Graph):
    def __init__(self, eps=1e-2):

        self.id = 2
        self.A = np.array([[0, 0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 1],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]], dtype=np.int16)

        self.pos = np.array([[0.0, 0.0],
                             [0.0, 1.0],
                             [0.5, 0.5],
                             [0.5 + np.sqrt(2) / 2, 0.5],
                             [1.0 + np.sqrt(2) / 2, 0.0],
                             [1.0 + np.sqrt(2) / 2, 1.0]])

        # Set boundaries
        tmin = 0.
        tmax = 10.
        xmin = 0.
        xmax = 1.

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        self.dirichletNodes = np.array([0, 1, 4, 5])
        self.dirichletAlpha = np.array([0.9, 0.3, 0., 0., 0., 0.])
        self.dirichletBeta = np.array([0.0, 0.0, 0., 0., 0.8, 0.1])

        self.eps = eps

    def f(self, u):
        return u * (1 - u)

    def df(self, u):
        return 1 - 2 * u

    def pde(self, u, ut, ux, uxx):
        return ut - self.eps * uxx + self.df(u) * ux

    def flux(self, u, ux):
        return - self.eps * ux + self.f(u)

    def initial_cond(self, x):
        return np.zeros_like(x)


class Example3(Graph):
    def __init__(self, eps=1e-2):

        self.id = 3
        self.A = np.array([[0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]], dtype=np.int16)

        self.pos = np.array([[0, 0], [1, 0], [2, 0]])

        # Set boundaries
        tmin = 0.
        tmax = 10.
        xmin = 0.
        xmax = 1.

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        self.dirichletNodes = np.array([0, 2])
        self.dirichletAlpha = np.array([0.7, 0.0, 0.0])
        self.dirichletBeta = np.array([0.0, 0.0, 0.8])

        self.eps = eps

    def f(self, u):
        return u * (1 - u)

    def df(self, u):
        return 1 - 2 * u

    def pde(self, u, ut, ux, uxx):
        return ut - self.eps * uxx + self.df(u) * ux

    def flux(self, u, ux):
        return - self.eps * ux + self.f(u)

    def initial_cond(self, x):
        return np.zeros_like(x)
