import matplotlib.animation as animation
import matplotlib

import numpy as np
import scipy as sp
import scipy.optimize
import matplotlib.pyplot as plt
# import sys
import tensorflow as tf
from time import time

import graph

from quantumGraphSolverFVM import QuantumGraphSolverFVM
from quantumGraphSolverPINN import TimesteppingPINNSolver
# from quantumGraphSolverPINNCont import TimesteppingPINNSolver

xth_frame = 5
# Time discretization of PINN
N_b = 200
# Time discretization of FVM
nt = N_b * xth_frame

# Spatial discretization of PINN
N_0 = 200
# Spatial discretization of FVM
nx = N_0 + 1


timestepmode = 'implicit'

printGraph = False
printCollocationPoints = True
gen_fvm_movie = True
gen_pinn_movie = True
gen_diff_movie = True

# graph = graph.Example0()
# graph = graph.Example1()
# graph = graph.Example2()
graph = graph.Example3()
graph.buildGraph()

if printGraph:
    graph.plotGraph()
    plt.show()
    plt.pause(0.01)

fvm_solver = QuantumGraphSolverFVM(graph)
u = fvm_solver.solve(nx=nx, nt=nt)

if gen_fvm_movie:
    matplotlib.use("Agg")
    print('Generate video...')
    fig = plt.figure(figsize=(6, 5))
    fig.subplots_adjust(left=0, bottom=0,
                        right=1, top=1,
                        wspace=None, hspace=None)

    def interactive_net(j=0):
        fvm_solver.plotNetwork(j=j * xth_frame, fig=fig)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15,
    #                 metadata=dict(artist='Jan Blechschmidt'),
    #                 bitrate=1800)
    writer = Writer(fps=15, bitrate=1800)

    line_ani = animation.FuncAnimation(fig, interactive_net,
                                       round(nt / xth_frame), interval=50)
    s = 'sol_fvm_{:d}.mp4'.format(graph.id)
    line_ani.save(s, writer=writer, dpi=200)
    print(s)

#### PINN stuff ####
t_r = tf.linspace(graph.lb[0], graph.ub[0], N_b + 1)
x_r = tf.linspace(graph.lb[1], graph.ub[1], N_0 + 1)
x_r = tf.reshape(x_r, (-1, 1))

print('Time step size: ', t_r[1].numpy() - t_r[0].numpy())

tf.random.set_seed(0)
pinn_solver = TimesteppingPINNSolver(graph, t_r, x_r)

# Solve problem
pinn_solver.ts_scheme()

if gen_pinn_movie:

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
    s = 'sol_pinn_{:d}.mp4'.format(graph.id)
    line_ani.save(s, writer=writer, dpi=200)
    print(s)

#### Prepare comparison video

if gen_diff_movie:

    matplotlib.use("Agg")
    print('Generate video...')
    fig = plt.figure(figsize=(6, 5))
    fig.subplots_adjust(left=0, bottom=0,
                        right=1, top=1,
                        wspace=None, hspace=None)

    pos = graph.pos
    E = graph.E
    L = graph.ub[1] - graph.lb[1]
    xy_list = [pos[e[0]] + x_r * (pos[e[1]] - pos[e[0]]) / L for e in E]

    def plot_diff_Network(j=0, fig=None):
        if fig is None:
            fig = plt.figure(1, clear=True)
        else:
            fig.clf()

        ax = fig.add_subplot(1, 1, 1, projection='3d')

        for i, e in enumerate(E):
            pinnu = pinn_solver.U[i][j, :]
            fvmu = fvm_solver.get_u_edge(fvm_solver.u, i, j * xth_frame)
            ax.plot(xy_list[i][:, 0], xy_list[i][:, 1], np.abs(fvmu - pinnu))

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlim([.0, 1.0])
        # ax.view_init(12, 135)
        ax.view_init(12, 290)

    def interactive_net(j=0):
        plot_diff_Network(j=j, fig=fig)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15,
    #                 metadata=dict(artist='Jan Blechschmidt'),
    #                 bitrate=1800)
    writer = Writer(fps=15, bitrate=1800)

    line_ani = animation.FuncAnimation(fig, interactive_net,
                                       N_b, interval=50)
    s = 'sol_diff_{:d}.mp4'.format(graph.id)
    line_ani.save(s, writer=writer, dpi=200)
    print(s)
