import numpy as np
from model import Problem
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
                         param_type=param_type) 
        self.multi_dims = multi_dims

        self.contour_density = None
        self.colorbar = None

    ## Overide Methods ##
    def _plot(self, 
              result=None,
              plot_3D=False, 
              contour_density=25,
              colorbar=False, 
              show_pareto_front=True,
              points=None,
              **kwargs):
        if self.n_params != 2:
            raise Exception('Cannot plot problem with more than 2 dimensions!')
        self.colorbar = colorbar
        self._points = points
        self.contour_density = contour_density
        self.initialize_plot(plot_3D=plot_3D, 
                             xlabel=r'$x_1$', 
                             ylabel=r'$x_2$')
        ax = self.contour3D() if plot_3D else self.contour2D()
        if result is not None:
            X, Y = result.history[-1]['P'][:, 0], result.history[-1]['P'][:, 1]
            Z = result.history[-1]['F'] if plot_3D else None
            self.scatter(X, Y, Z)
        if self.colorbar:
            self.fig.colorbar(ax)
        plt.show()
        self.ax.clear()

    def _make_data(self):
        
        (xl, xu) = self.domain
        self._points = 100 if self._points is None else self._points
        axis_points = np.linspace(xl, xu, self._points)
        n = len(axis_points)
        if n % 2 != 0:
            axis_points = axis_points[:-1]
        x_mesh, y_mesh = np.meshgrid(axis_points, axis_points)

        f = self._f
        if self.multi_dims:
            n = len(x_mesh)
            
            z_mesh = [f(np.array([x_mesh[i], y_mesh[i]])) for i in range(n)]
            z_mesh = np.array(z_mesh)
        else:
            z_mesh = np.array(f([x_mesh, y_mesh]))

        self._data = (x_mesh, y_mesh, z_mesh)

    ## Public Methods ##
    def contour2D(self):
        (X, Y, Z) = self._data
        global_optimums = self._pareto_set
        if type(global_optimums) != type(None):
            global_optimums = global_optimums.reshape((-1, 2))
            self.ax.plot(global_optimums[:, 0],
                         global_optimums[:, 1],
                         'rx',
                         label='global optimum',
                         markersize=10)
        self.ax.grid(True, linestyle='--')
        ax = self.ax.contour(X, Y, Z, 
                            self.contour_density,
                            cmap=cm.jet)
        
        return ax


    def contour3D(self):
        (X, Y, Z) = self._data
        surf = self.ax.plot_surface(X, Y, Z, 
                                    cmap=cm.jet,
                                    rstride=1,
                                    cstride=1,
                                    linewidth=0.2)
        self.ax.set_zlabel(r'$f(x_1, x_2)$')
        return surf

    def scatter(self, X, Y, Z):
        if Z is None:
            lim = (self.ax.get_xlim(), self.ax.get_ylim())
            self.ax.plot(X, Y, 'g.', label='genome')
            self.ax.set_xlim(lim[0])
            self.ax.set_ylim(lim[1])
        else:
            lim = (self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim())
            self.ax.scatter(X, Y, Z, 'g.', label='genome')
            self.ax.set_xlim(lim[0])
            self.ax.set_ylim(lim[1])
            self.ax.set_zlim(lim[2])
        self.ax.legend(loc='upper right')
