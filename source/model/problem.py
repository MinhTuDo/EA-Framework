import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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

    def evaluate_all(self, pop):
        f_pop = np.array(list(map(self._function, pop)))
        return f_pop

    def plot(self, plot_3D=False):
        if self.n_params != 2:
            raise Exception('Cannot plot problem with more than 2 dimensions!')
        if plot_3D:
            self._contour3D()
        else:
            self.__contour2D()
        pass

    def get_plot_data(self):
        step = 1 if self.param_type == np.int else 0.1
        (xl, xu) = self.domain
        if self.problem.param_type == np.int:
            xu += 1
        axis_points = np.arange(xl, xu, step)
        n = len(axis_points)
        if n % 2 != 0:
            axis_points = axis_points[:-1]
        x_mesh, y_mesh = np.meshgrid(axis_points, axis_points)

        f = self.problem._function
        if self.problem.multi_dims:
            n = len(x_mesh)
            
            z_mesh = [f(np.array([x_mesh[i], y_mesh[i]])) for i in range(n)]
            z_mesh = np.array(z_mesh)
        else:
            z_mesh = np.array(f([x_mesh, y_mesh]))

        self.problem_mesh = (x_mesh, y_mesh, z_mesh)
        pass


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
    def __contour2D(self):
        pass

    def __contour3D(self):
        pass