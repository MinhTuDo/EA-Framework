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

        self.temp_dir = 'temp/'
        self.__make_dir(self.temp_dir)

        self.data = None
        if contour_density is None:
            self.contour_density = 25
        else:
            self.contour_density = contour_density

        self.loc = None

        self.fig, self.ax = self.problem.fig, self.problem.ax
        self.axis_labels = None
        self.display_optimum = None
        self.problem_space_lim = None


    def create_problem_space(self): 
        self.__get_data()

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
        if self.data is None:
            self.__get_data()
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
        os.system('convert {}*.png {}.gif'.format(self.temp_dir, 
                                                   filepath_temp))

    def __animate(self):
        problem_contour = self.problem.contour_density
        self.problem.contour_density = self.contour_density
        for gen, res in enumerate(self.result.history):
            self.ax.clear()
            self.__contour()
            self.__scatter(res['P'][:, 0], res['P'][:, 1], gen+1)
            name = '{:0>3d}.png'.format(gen)
            self.fig.suptitle(self.title[0], fontsize=self.title[1])
            self.fig.savefig(os.path.join(self.temp_dir, name))
        
        self.problem.contour_density = problem_contour
            

    def __contour(self):
        self.problem.contour2D(self.display_optimum)
        self.ax.set_xlabel(self.axis_labels[0])
        self.ax.set_ylabel(self.axis_labels[1])
        self.problem_space_lim = (self.ax.get_xlim(), self.ax.get_ylim())

    def __scatter(self, X, Y, gen):
        self.ax.plot(X, Y, 'g.', label='genome')

        (xlim, ylim) = self.problem_space_lim
        self.ax.legend(loc=self.loc)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        title = '{} at Gen {}'.format(self.problem.__class__.__name__,gen)
        self.ax.set_title(title)

    def __get_data(self):
        if self.problem.data is None:
            self.problem.make_data()
        self.data = self.problem.data
