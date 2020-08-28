from model.problem import Problem
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "serif"

class MultiObjectiveProblem(Problem):
    def __init__(self, 
                 n_params=2, 
                 n_obj=2,
                 n_constraints=0,
                 param_type=None):
        super().__init__(n_params=n_params,
                         n_obj=n_obj,
                         n_constraints=n_constraints,
                         param_type=param_type)
        
        self._argopt = self.arg_optimum
        self._optimum = self.optimum

    ## Protected Methods ##
    def _is_dominated(self, y1, y2):
        pass
    
    def _calc_pareto_front(self):
        pass

    ## Overide Methods ##
    def _sol_compare(self, s1, s2):
        r1, r2 = s1[0], s2[0]
        cd1, cd2 = s1[1], s2[1]
        return (r1 < r2) or \
               (r1 == r2 and cd1 > cd2)

    def _plot(self,
              result=None,
              points=None,
              **kwargs):
        if self.n_obj > 3:
            raise Exception('Cannot plot problem with more than 3 objectives')
        plot_3D = True if self.n_obj == 3 else False
        self.initialize_plot(plot_3D=plot_3D,
                             xlabel=r'$f_1(x)$',
                             ylabel=r'$f_2(x)$')
        self.plot_3D() if self.n_obj == 3 else self.plot_2D()
        if result is not None:
            X, Y = result.history[-1]['F'][:, 0], result.history[-1]['F'][:, 1]
            Z = result.history[-1]['F'][:, 2] if plot_3D else None
            self.scatter(X, Y, Z)
        self.ax.legend(loc='upper right')
        plt.show()

    def initialize_plot(self, **kwargs):
        xlabel, ylabel = r'$f(x_1)$', r'$f(x_2)$'
        self.fig, self.ax = plt.subplots()
        if self.n_obj == 3:
            self.ax = Axes3D(self.fig, azim=-29, elev=50)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, linestyle='--')
        plt.suptitle('{}'.format(self.__class__.__name__))
        self._make_data()

    def _make_data(self):
        self._points = 100 if self._points is None else self._points
        self._data = self._calc_pareto_front()


    ## Public Methods ##
    def plot_2D(self):
        if self._data is not None:
            (X, Y) = self._data
            if Y is not None:
                self.ax.plot(X, Y, label='pareto front', color='red')
                self.ax.grid(True, linestyle='--')

    def plot_3D(self):
        if self._data is not None:
            (X, Y, Z) = self._data
            if Z is not None:
                self.ax.plot(X, Y, Z, label='pareto front', color='red')

    def scatter(self, X, Y, Z):
        if Z is None:
            lim = (self.ax.get_xlim(), self.ax.get_ylim())
            self.ax.plot(X, Y, 'g.', label='genome')
            # self.ax.set_xlim(lim[0])
            # self.ax.set_ylim(lim[1])
        else:
            lim = (self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim())
            # self.ax.scatter(X, Y, Z, 'g.', label='genome')
            # self.ax.set_xlim(lim[0])
            # self.ax.set_ylim(lim[1])
            # self.ax.set_zlim(lim[2])
        self.ax.legend(loc='upper right')


    def optimum(self, Y):
        opt = Y[0]
        for y in Y:
            opt = opt if self._sol_compare(opt, y) else y
        return opt

    def arg_optimum(self, Y):
        argopt = 0
        for i, y_i in enumerate(Y):
            argopt = argopt if self._sol_compare(Y[argopt], y_i) else i
        return argopt
    