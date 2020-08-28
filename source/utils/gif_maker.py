import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

class GifMaker:
    def __init__(self,
                 directory,
                 filename,
                 loop=None,
                 title=None,
                 **kwargs):
        self.directory = directory
        self.filename = filename
        self.title = title

        self.temp_dir = './temp/'
        self.make_dir(self.temp_dir)
        self.make_dir(self.directory)

        self.avail_loc = ['upper left', 
                          'upper right',
                          'lower left', 
                          'lower right']
        self.loc = None

        self._fig, self._ax = None, None
        self._plot_3D = None
        

    def make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def convert_png_to_gif(self):
        filepath_temp = os.path.join(self.temp_dir, self.filename)
        command = 'convert -loop 1 {}*.png {}.gif'.format(self.temp_dir, 
                                                          filepath_temp)
        os.system(command)

    def get_data(self):
        self.problem.initialize_plot(_plot_3D=self._plot_3D)
        self.data = self.problem._data

    def make(self,
            result,   
            loc='upper right',
            **kwargs):
        self._make(**kwargs)
        self.result = result
        self.loc = loc
        self.get_data()
        self._ax, self._fig = self.problem.ax, self.problem.fig
        print('Processing...')
        self.animate()
        self.convert_png_to_gif()
        
        temp_path = os.path.join(self.temp_dir, '{}.gif'.format(self.filename))
        os.system('mv {} {}'.format(temp_path, self.directory))
        print('Done!')

        ## Remove temporary files
        os.system('rm {}*'.format(self.temp_dir))

    def animate(self):
        plot = self._animate()
        for gen, res in enumerate(self.result.history):
            self.make_plot(plot, res, gen)
            self.save_plot(gen)

    def make_plot(self, plot, res, gen):
        X, Y, Z = self._make_plot(plot, res, gen)
        self.problem.scatter(X, Y, Z)
        title = '{} at Gen {}'.format(self.problem.__class__.__name__, gen+1)
        self._ax.set_title(title)
    
    def save_plot(self, gen):
        name = '{:0>3d}.png'.format(gen+1)
        self._fig.suptitle(self.title[0], fontsize=self.title[1])
        self._fig.savefig(os.path.join(self.temp_dir, name))
        self._ax.clear()

    def _make(self, **kwargs):
        pass

    def _make_plot(self, plot, res, gen):
        pass


class SOGifMaker(GifMaker):
    def __init__(self, 
                 problem,
                 filename,
                 directory='gif', 
                 contour_density=None,
                 title=('', 12),
                 **kwargs):

        super().__init__(directory=directory,
                         filename=filename,
                         title=title,
                         **kwargs)
        if problem.n_params > 2:
            raise Exception('Cannot plot problem with more than 2 dimensions!')
        if problem.n_obj > 1:
            raise Exception('Cannot plot problem with more than 1 objective')
        self.problem = problem

        self.data = None
        self.contour_density = 25 if contour_density is None else contour_density

    def _make(self, **kwargs):
        self._plot_3D = kwargs.get('_plot_3D')


    def _animate(self):
        problem_contour = self.problem.contour_density
        self.problem.contour_density = self.contour_density
        contour = self.problem.contour2D if not self._plot_3D else self.problem.contour3D
        return contour
        

    def _make_plot(self, contour, res, gen):
        contour()
        self._ax.set_xlabel(r'$x_1$')
        self._ax.set_ylabel(r'$x_2$')
        X, Y = res['P'][:, 0], res['P'][:, 1]
        Z = None if not self._plot_3D else res['F']
        return X, Y, Z
        

class MOGifMaker(GifMaker):
    def __init__(self,
                 problem,
                 filename,
                 directory='gif',
                 title=('', 12),
                 **kwargs):
        
        super().__init__(directory=directory,
                         filename=filename,
                         title=title,
                         **kwargs)
        if problem.n_obj > 3:
            raise Exception('Cannot plot problem with more than 1 objective')
        self.problem = problem

        self.data = None
        self._plot_3D = self.problem.n_obj == 3

    def _animate(self):
        plot = self.problem.plot_2D if not self._plot_3D else self.problem._plot_3D
        return plot

    def _make_plot(self, plot, res, gen):
        plot()
        self._ax.set_xlabel(r'$f_1(x)$')
        self._ax.set_ylabel(r'$f_2(x)$')
        X, Y = res['F'][:, 0], res['F'][:, 1]
        Z = None if not self._plot_3D else res['F'][:, 2]
        return X, Y, Z
        

        
        


