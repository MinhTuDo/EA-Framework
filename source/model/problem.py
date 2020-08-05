import numpy as np
import matplotlib
matplotlib.use('tkagg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker
import matplotlib.pyplot as plt

class Problem:
    def __init__(self,
                 n_params=-1,
                 n_obj=-1,
                 n_constraints=0,
                 domain=(0, 0),
                 param_type=None,
                 multi_dims=False):
        if n_params <= 0:
            raise Exception('Parameters length must be greater than zero')
        self.n_params = n_params
        self.n_obj = n_obj
        self.n_constraints = n_constraints
        self.domain = domain
        self.type = param_type
        self.multi_dims = multi_dims
        self.param_type = param_type

        self._pareto_front = None
        self._pareto_set = None
        self._optimum = None
        self._argopt = None

        self.data = None
        self.contour_density = None
        self.fig, self.ax = None, None
        self.colorbar = None
        self.plot_3D = None
        self.step = None

    def evaluate_all(self, pop):
        f_pop = np.array(list(map(self._function, pop)))
        return f_pop

    def plot(self, 
             plot_3D=False, 
             contour_density=25,
             colorbar=False, 
             display_optimum=True):
        if self.n_params != 2:
            raise Exception('Cannot plot problem with more than 2 dimensions!')
        self.plot_3D = plot_3D
        self.colorbar = colorbar
        self.contour_density = contour_density
        self.initialize_plot()
        ax = None
        if plot_3D:
            ax = self.contour3D()
        else:
            ax = self.contour2D()

        if self.colorbar:
            self.fig.colorbar(ax)
        plt.show()

    def make_data(self):
        if self.step is None:
            self.step = 1 if self.param_type == np.int else 0.1
        (xl, xu) = self.domain
        if self.param_type == np.int:
            xu += 1
        axis_points = np.arange(xl, xu, self.step)
        n = len(axis_points)
        if n % 2 != 0:
            axis_points = axis_points[:-1]
        x_mesh, y_mesh = np.meshgrid(axis_points, axis_points)

        f = self._function
        if self.multi_dims:
            n = len(x_mesh)
            
            z_mesh = [f(np.array([x_mesh[i], y_mesh[i]])) for i in range(n)]
            z_mesh = np.array(z_mesh)
        else:
            z_mesh = np.array(f([x_mesh, y_mesh]))

        self.data = (x_mesh, y_mesh, z_mesh)


    ## Protected Methods ##
    def _function(self, X):
        pass

    def _pareto_front(self):
        pass

    def _pareto_set(self):
        pass

    def _comparer(self, x1, x2):
        pass
    def _f_comparer(self, y1, y2):
        pass

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