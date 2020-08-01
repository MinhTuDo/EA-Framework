import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class GifSaver:
    def __init__(self, 
                 problem,
                 directory, 
                 filename,
                 contour_density=None,
                 title=('', 12)):
        if problem.n_params > 2:
            raise Exception('Cannot plot problem with more than 2 dimensions!')
        self.problem = problem
        self.directory = directory
        self.__make_dir(self.directory)
        self.filename = filename

        self.avail_loc = ['upper left', 'upper right',
                          'lower left', 'lower right']
        self.markersize = 10
        self.title = title

        self.temp_dir = './temp/'
        self.__make_dir(self.temp_dir)

        self.problem_mesh = None
        if contour_density is None:
            self.contour_density = 25
        else:
            self.contour_density = contour_density

        self.loc = None

        self.fig, self.ax = plt.subplots()
        self.axis_labels = None
        self.display_optimum = None
        self.problem_space_lim = None


    def create_problem_space(self): 
        self.__get_problem_mesh()

    def make(self,
            result,
                
            axis_labels=['x1', 'x2'],
            legend=None,
            loc='upper right',
            display_optimum=False):
        self.result = result
        self.axis_labels=axis_labels
        self.legend=legend
        self.loc = loc
        self.display_optimum = display_optimum
        if self.problem_mesh is None:
            self.__get_problem_mesh()
        print('Processing...')
        self.__animate()
        self.__convert_png_to_gif()
        
        temp_path = os.path.join(self.temp_dir, '{}.gif'.format(self.filename))
        os.system('mv {} {}'.format(temp_path, self.directory))
        print('Done!')

        ## Remove temporary files
        os.system('rm {}*'.format(self.temp_dir))

    def __make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def __convert_png_to_gif(self):
        filepath_temp = os.path.join(self.temp_dir, self.filename)
        os.system('convert {}g*.png {}.gif'.format(self.temp_dir, 
                                                   filepath_temp))

    def __animate(self):
        for gen, res in enumerate(self.result.history):
            self.ax.clear()
            self.__contour()
            self.__scatter(res['P'][:, 0], res['P'][:, 1], gen+1)
            name = 'g{:0>3d}.png'.format(gen)
            self.fig.suptitle(self.title[0], fontsize=self.title[1])
            self.fig.savefig(os.path.join(self.temp_dir, name))
            

    def __contour(self):
        (X, Y, Z) = self.problem_mesh
        global_optimums = self.problem._pareto_set
        if type(global_optimums) != type(None) and self.display_optimum:
            global_optimums = global_optimums.reshape((-1, 2))
            self.ax.plot(global_optimums[:, 0],
                        global_optimums[:, 1],
                        'rx',
                        label='global optimum',
                        markersize=self.markersize)
        
        self.ax.contour(X, Y, Z, 
                        self.contour_density,
                        cmap=cm.jet)
        self.ax.set_xlabel(self.axis_labels[0])
        self.ax.set_ylabel(self.axis_labels[1])
        self.ax.grid(True)

        self.problem_space_lim = (self.ax.get_xlim(), 
                                  self.ax.get_ylim())

    def __scatter(self, X, Y, gen):
        self.ax.plot(X, Y, 'g.', label='genome')

        (xlim, ylim) = self.problem_space_lim
        self.ax.legend(loc=self.loc)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        title = '{} at Gen {}'.format(self.problem.__class__.__name__,gen)
        self.ax.set_title(title)

    def __get_problem_mesh(self):
        step = 1 if self.problem.param_type == np.int else 0.1
        (xl, xu) = self.problem.domain
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