import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
import scipy
from time import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import graph

DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)


class PINN(tf.keras.Model):
    """ Set basic architecture of the PINN model."""

    def __init__(self, lb, ub,
                 output_dim=1,
                 num_hidden_layers=3,
                 num_neurons_per_layer=20,
                 activation='tanh',
                 kernel_initializer='glorot_normal',
                 **kwargs):

        super().__init__(**kwargs)

        self.n = num_hidden_layers
        self.input_dim = lb.shape[0]
        self.output_dim = output_dim
        self.lb = lb
        self.ub = ub

        # Define NN architecture

        # Scaling layer to map input to the interval [-1, 1]
        # self.scale = tf.keras.layers.Lambda(
        #    lambda x: 2.0 * (x - lb) / (ub - lb) - 1.0)

        # Inititialize n many fully connected dense layers
        act = tf.keras.activations.get(activation)

        def _layer():
            return Dense(num_neurons_per_layer,
                         activation=act,
                         kernel_initializer=kernel_initializer)

        self.hidden = [_layer() for _ in range(self.n)]

        # Output layer
        # self.out = Dense(output_dim, activation=None)
        self.out = Dense(output_dim, activation='sigmoid')

    def call(self, X):
        """Forward-pass through neural network."""

        # Z = self.scale(X)
        Z = X

        for i in range(self.n):
            Z = self.hidden[i](Z)

        return self.out(Z)


class TimesteppingPINNSolver(object):
    def __init__(self, graph, t_r, x_r):

        self.graph = graph
        self.ne = self.graph.ne

        self._setupNNs()

        self.t = t_r
        self.x = x_r
        self.dt = t_r[1].numpy() - t_r[0].numpy()

        self.nx = x_r.shape[0]
        self.nt = t_r.shape[0]

        # Initialize history of losses and global iteration counter
        self.hist = []
        self.iter = 0
        self.current_loss = 999.

        self.idx = 1

        self.U = []
        self.uold = []
        for i in range(self.ne):
            self.uold.append(tf.Variable(
                self.graph.initial_cond(self.x)[:, 0]))
            self.U.append(1e5 * np.ones(shape=(self.nt, self.nx)))
            self.U[i][0, :] = self.uold[i].numpy()

        # Call each network once to initialize trainable variables
        self.trainable_variables = []
        for i in range(self.ne):
            self.NNs[i](tf.constant([[1.]], dtype=DTYPE))
            self.trainable_variables.append(self.NNs[i].trainable_variables)

        # Setup auxiliary variables for vertex values to ensure continuity
        self._setupVertexVariables()

        for i, v in enumerate(self.graph.innerVertices):
            self.trainable_variables.append([self.vertexVals[i]])

        self.nvar = len(self.trainable_variables)

    def _setupNNs(self):

        self.NNs = []
        for i, e in enumerate(self.graph.E):
            self.NNs.append(PINN(lb=self.graph.lb, ub=self.graph.ub))

        print('Initialized {:d} neural nets.'.format(len(self.NNs)))

    def _setupVertexVariables(self):

        self.vertexVals = []
        for _ in self.graph.innerVertices:
            vvar = tf.Variable(
                tf.random.uniform(
                    shape=(1,), dtype=DTYPE),
                trainable=True)
            self.vertexVals.append(vvar)

    def _fvals0(self, x):

        # Initialize lists for values and derivatives
        u = []
        for i in range(self.ne):
            u.append(self.NNs[i](x)[:, 0])

        return u

    def _fvals1(self, x):

        # Initialize lists for values and derivatives
        u = []
        ux = []

        for i in range(self.ne):

            with tf.GradientTape(persistent=True) as tape:
                # Watch variables representing t and x during this GradientTape
                tape.watch(x)

                # Compute current values u(t,x)
                u.append(self.NNs[i](x)[:, 0])
            ux.append(tape.gradient(u[i], x)[:, 0])

            del tape

        return u, ux

    def _fvals2(self, x):

        # Initialize lists for values and derivatives
        u = []
        ux = []
        uxx = []

        for i in range(self.ne):

            with tf.GradientTape(persistent=True) as tape:
                # Watch variables representing t and x during this GradientTape
                tape.watch(x)

                # Compute current values u(t,x)
                u.append(self.NNs[i](x)[:, 0])
                ux.append(tape.gradient(u[i], x)[:, 0])

            uxx.append(tape.gradient(ux[i], x)[:, 0])

            del tape

        return u, ux, uxx

    def determine_losses(self):

        # Short-hand notation of mean-squared loss
        def mse(x):
            return tf.reduce_mean(tf.square(x))

        ###################################
        ### Residual loss for all edges ###
        ###################################
        u, ux, uxx = self._fvals2(self.x)

        loss_res = 0
        for i in range(self.ne):
            res_e = u[i] - self.uold[i] + self.dt * \
                self.graph.pde(u[i], 0., ux[i], uxx[i])

            #print(self.U[i][self.idx-1, :])
            #loss_res += mse(res_e[1:-1])
            loss_res += mse(res_e)

            # print(mse(res_e))
            #print(self.U[i][self.idx-1, :])

        ###################################
        ###   Continuity in vertices    ###
        ###################################

        # ul, ult, ulx = self._fvals1(self.Xl[:,0], self.Xl[:,1])
        # uu, uut, uux = self._fvals1(self.Xu[:,0], self.Xu[:,1])
        loss_cont = 0

        for i, v in enumerate(self.graph.innerVertices):

            for j in self.graph.Vin[v]:
                val = u[j][-1] - self.vertexVals[i]
                loss_cont += mse(val)

            for j in self.graph.Vout[v]:
                val = u[j][0] - self.vertexVals[i]
                loss_cont += mse(val)

        #####################################
        ### Kirchhoff-Neumann in vertices ###
        #####################################

        # Kirchhoff-Neumann condition in center nodes
        loss_KN = 0
        for i in self.graph.innerVertices:

            val = 0
            #print('Kirchhoff-Neumann in node ', i)
            for j in self.graph.Vin[i]:
                #print('incoming edge:', j)
                val += self.graph.flux(u[j][-1], ux[j][-1])
                #val += self.graph.flux(uu[j], uux[j])

            for j in self.graph.Vout[i]:
                #print('outgoing edge:', j)
                val -= self.graph.flux(u[j][0], ux[j][0])
                #val -= self.graph.flux(ul[j], ulx[j])
            loss_KN += mse(val)

        #####################################
        ###      Inflow/Outflow conds     ###
        #####################################

        loss_D = 0
        for i, v in enumerate(self.graph.dirichletNodes):

            alpha = self.graph.dirichletAlpha[v]
            beta = self.graph.dirichletBeta[v]

            print('\nin node ', v, 'alpha ', alpha, 'beta ', beta)
            val = 0
            #print('\n', val)
            for j in self.graph.Vin[v]:
                print('outflow: ', j)
                # val += -self.graph.flux(uu[j], uux[j]) - beta * (uu[j])
                val += self.graph.flux(u[j][-1], ux[j][-1]) - beta * (u[j][-1])
                # loss_D += mse(val)
            # print(val)

            for j in self.graph.Vout[v]:
                print('inflow: ', j)
                # val += -self.graph.flux(ul[j], ulx[j]) + alpha * (1-ul[j])
                val += -self.graph.flux(u[j][0], ux[j][0]) + \
                    alpha * (1. - u[j][0])
                # val += -self.graph.flux(ul[j], ulx[j]) + alpha * (1-ul[j])
                # loss_D += mse(val)
            # print(val, '\n')
            loss_D += mse(val)

        return loss_res, loss_cont, loss_KN, loss_D

    def loss_fn(self):

        loss_res, loss_cont, loss_KN, loss_D = self.determine_losses()

        loss = loss_res + loss_cont + loss_KN + loss_D
        # print(loss_res)
        # print(loss_cont)
        # print(loss_KN)
        # print(loss_D)
        # print(loss)
        return loss

    @tf.function
    def get_grad(self):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.trainable_variables)
            loss = self.loss_fn()

        g = tape.gradient(loss, self.trainable_variables)
        del tape

        return loss, g

    def solve_with_TFoptimizer(self, optimizer, eps, N=1001):
        """This method performs a gradient descent type optimization."""

        self.callback_init()

        for i in range(N):
            loss, g = self.get_grad()
            # Perform gradient descent step
            for j in range(self.nvar):
                optimizer.apply_gradients(
                    zip(g[j], self.trainable_variables[j]))
            # print(self.current_loss)
            self.current_loss = loss.numpy()
            # print(self.current_loss)
            # print(loss)
            # print('\n')

            self.callback()

            if self.current_loss < eps:
                break

    def solve_with_ScipyOptimizer(self, method='L-BFGS-B', **kwargs):
        """This method provides an interface to solve the learning problem
        using a routine from scipy.optimize.minimize.
        (Tensorflow 1.xx had an interface implemented, which is not longer
        supported in Tensorflow 2.xx.)
        Type conversion is necessary since scipy-routines are written in
        Fortran which requires 64-bit floats instead of 32-bit floats."""

        def get_weight_tensor():
            """Function to return current variables of the model
            as 1d tensor as well as corresponding shapes as lists."""

            weight_list = []
            shape_list = []

            # Loop over all variables, i.e. weight matrices, bias vectors
            # and unknown parameters
            for i in range(len(self.trainable_variables)):
                for v in self.trainable_variables[i]:
                    shape_list.append(v.shape)
                    weight_list.extend(v.numpy().flatten())

            weight_list = tf.convert_to_tensor(weight_list)
            return weight_list, shape_list

        x0, shape_list = get_weight_tensor()

        def set_weight_tensor(weight_list):
            """Function which sets list of weights
            to variables in the model."""
            idx = 0

            for i in range(len(self.trainable_variables)):
                for v in self.trainable_variables[i]:
                    vs = v.shape

                    # Weight matrices
                    if len(vs) == 2:
                        sw = vs[0] * vs[1]
                        new_val = tf.reshape(
                            weight_list[idx:idx + sw], (vs[0], vs[1]))
                        idx += sw

                    # Bias vectors
                    elif len(vs) == 1:
                        new_val = weight_list[idx:idx + vs[0]]
                        idx += vs[0]

                    # Variables (in case of parameter identification setting)
                    elif len(vs) == 0:
                        new_val = weight_list[idx]
                        idx += 1

                    # Assign variables (Casting necessary since scipy requires float64 type)
                    v.assign(tf.cast(new_val, DTYPE))

        def get_loss_and_grad(w):
            """Function that provides current loss and gradient
            w.r.t the trainable variables as vector. This is mandatory
            for the LBFGS minimizer from tfp.optimizer."""

            # Update weights in model
            set_weight_tensor(w)
            # Determine value of \phi and gradient w.r.t. \theta at w
            loss, grad = self.get_grad()
            # Flatten gradient
            grad_flat = []
            for i in range(len(self.trainable_variables)):
                for g in grad[i]:
                    grad_flat.extend(g.numpy().flatten())

            # Store current loss for callback function
            # print(self.current_loss)
            self.current_loss = loss
            # print(self.current_loss)

            # Return value and gradient of \phi as tuple
            return loss.numpy().astype(np.float64), np.array(grad_flat, dtype=np.float64)

        self.callback_init()

        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       callback=self.callback,
                                       **kwargs)

    def ts_scheme(self, eps=1e-6):
        max_trials = 1
        while self.idx < self.nt:
            print('Solve time step {}/{}\n'.format(self.idx, self.nt))

            trial = 0
            while trial < max_trials:

                print('Adam...\n')

                if self.idx == 1:
                    lr = 0.01
                    optim = tf.keras.optimizers.Adam(learning_rate=lr)
                    self.solve_with_TFoptimizer(optim, eps=eps, N=2001)

                else:
                    lr = 0.001
                    optim = tf.keras.optimizers.Adam(learning_rate=lr)
                    self.solve_with_TFoptimizer(optim, eps=eps, N=301)
                self.callback(force=True)
                print('LBFGS...\n')
                ret = self.solve_with_ScipyOptimizer(
                    options={'maxiter': 50000,
                             'maxfun': 50000,
                             'maxcor': 50,
                             'maxls': 50,
                             'eps': eps,
                             'ftol': 1.0e3 * np.finfo(float).eps,
                             'gtol': 1.0e3 * np.finfo(float).eps})
                # factr is 10000000
                print(ret.message)
                trial += 1
                self.callback(force=True)

            u = self._fvals0(self.x)
            self.assign_u(self.idx, u)
            self.idx += 1
            self.iter = 0

        return self.U

    def assign_u(self, timestep, u):

        for i in range(self.ne):
            self.uold[i].assign(u[i])
            self.U[i][timestep, :] = u[i]

    def callback_init(self):
        self.t0 = time()
        print(' Iter            Loss    Time')
        print('-----------------------------')

    def callback(self, xr=None, force=False):
        if self.iter % 100 == 0 or force:
            print('{:05d}  {:10.8e}   {:4.2f}'.format(
                self.iter, self.current_loss, time() - self.t0))
        self.hist.append(self.current_loss)
        self.iter += 1

    def plot_loss_history(self, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.hist)), self.hist, 'k-')
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('$\\phi^{n_{epoch}}$')
        return ax

    def plotNetwork(self, j=0, u=None, fig=None):

        pos = self.graph.pos
        X = self.x
        E = self.graph.E
        L = self.graph.ub[1] - self.graph.lb[1]
        xy_list = [pos[e[0]] + X * (pos[e[1]] - pos[e[0]]) / L for e in E]

        if fig is None:
            fig = plt.figure(1, clear=True)
        else:
            fig.clf()

        if u is None:
            u = self.U

        ax = fig.add_subplot(1, 1, 1, projection='3d')

        for i, e in enumerate(self.graph.E):
            uij = u[i][j, :]
            ax.plot(xy_list[i][:, 0], xy_list[i][:, 1], uij)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlim([.0, 1.0])
        # ax.view_init(12, 135)
        ax.view_init(12, 290)

        return ax

if __name__ == '__main__':

    # Time discretization of PINN
    N_b = 200
    # Spatial discretization of PINN
    N_0 = 200

    printGraph = False
    generateMovie = True

    graph = graph.Example2()
    graph.buildGraph()

    if printGraph:
        graph.plotGraph()
        plt.show()
        plt.pause(0.01)

    t_r = tf.linspace(graph.lb[0], graph.ub[0], N_b + 1)
    x_r = tf.linspace(graph.lb[1], graph.ub[1], N_0 + 1)
    x_r = tf.reshape(x_r, (-1, 1))
    
    print('Time step size: ', t_r[1].numpy() - t_r[0].numpy())
    
    tf.random.set_seed(0)
    pinn_solver = TimesteppingPINNSolver(graph, t_r, x_r)

    # Solve problem
    pinn_solver.ts_scheme()

    if generateMovie:
        matplotlib.use("Agg")
        print('Generate video...')
        fig = plt.figure(figsize=(6, 5))
        fig.subplots_adjust(left=0, bottom=0,
                            right=1, top=1,
                            wspace=None, hspace=None)
    
        def interactive_net(j=0):
            pinn_solver.plotNetwork(j=j, fig=fig)
    
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=15,
        #                 metadata=dict(artist='Jan Blechschmidt'),
        #                 bitrate=1800)
        writer = Writer(fps=15, bitrate=1800)
    
        line_ani = animation.FuncAnimation(fig, interactive_net,
                                           N_b, interval=50)
        s = 'sol_pinn_cont_{:d}.mp4'.format(graph.id)
        line_ani.save(s, writer=writer, dpi=200)
