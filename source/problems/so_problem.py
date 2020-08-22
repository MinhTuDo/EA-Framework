import numpy as np
from model.problem import Problem
import matplotlib
matplotlib.use('tkagg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker
import matplotlib.pyplot as plt

class SingleObjectiveProblem(Problem):
    def __init__(self,
                 n_params=-1,
                 n_constraints=0,
                 domain=(0, 0),
                 param_type=None,
                 multi_dims=False):
        super().__init__(n_params,
                         n_obj=1,
                         n_constraints=n_constraints,
                         domain=domain,
                         param_type=param_type,
                         multi_dims=multi_dims) 

        self.data = None
        self.contour_density = None
        self.fig, self.ax = None, None
        self.colorbar = None
        self.plot_3D = None
        self.step = None

    ## Private Methods ##
    def contour2D(self, display_optimum=True):
        (X, Y, Z) = self.data
        global_optimums = self._pareto_set
        if type(global_optimums) != type(None) and display_optimum:
            global_optimums = global_optimums.reshape((-1, 2))
            self.ax.plot(global_optimums[:, 0],
                         global_optimums[:, 1],
                         'rx',
                         label='global optimum',
                         markersize=10)
        self.ax.grid(True)
        ax = self.ax.contour(X, Y, Z, 
                            self.contour_density,
                            cmap=cm.jet)
        
        return ax


    def contour3D(self):
        (X, Y, Z) = self.data
        surf = self.ax.plot_surface(X, Y, Z, 
                                    cmap=cm.jet,
                                    rstride=1,
                                    cstride=1,
                                    linewidth=0.2)
        self.ax.set_zlabel("z")
        return surf

    def initialize_plot(self):
        xlabel, ylabel = 'x1', 'x2'
        self.fig, self.ax = plt.subplots()
        if self.plot_3D:
            self.ax = Axes3D(self.fig, azim=-29, elev=50)
            xlabel, ylabel = 'x', 'y'
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)
        plt.suptitle('{}'.format(self.__class__.__name__))
        self.make_data()
