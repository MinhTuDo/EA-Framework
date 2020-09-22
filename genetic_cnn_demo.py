from model import Display, LogSaver
from optimize import optimize
# from utils import SOGifMaker
from keras.utils import to_categorical
from keras.datasets import cifar10

from terminations import *

import problems.multi as mp

import operators.crossover as cx
import operators.initialization as init
import operators.mutation as mut
import operators.selection as sel
import operators.model_builder as mb

import algorithms.multi_objective as mo

import numpy as np
import time

class MySaver(LogSaver):
    def _do(self, ga):
        self.add_attributes('architecture', ga.pop[ga.elite_idx])
        self.add_attributes('error_rate', ga.F_pop[ga.elite_idx])
        self.add_attributes('flops', ga.F_pop[ga.elite_idx])
        self.add_attributes('n_gens', ga.n_gens) 
        self.add_attributes('n_evals', ga.n_evals) 
        
        now = time.time()
        self.add_attributes('time', now - ga.start_time)
        

class MyDisplay(Display):
    def _do(self, ga):
        self.display_top = -1
        self.add_attributes('n_gens', ga.n_gens) 
        self.add_attributes('n_evals', ga.n_evals) 
        self.add_attributes('error_rate', ga.F_pop[ga.elite_idx])
        self.add_attributes('flops', ga.F_pop[ga.elite_idx])
        now = time.time()
        self.add_attributes('time', now - ga.start_time)
        self.add_attributes('architecture', ga.pop[ga.elite_idx])
        

display = MyDisplay()
log_saver = MySaver()

NSGANET = mp.NSGANET
class CustomizedNSGANET(NSGANET):
    def _preprocess_input(self):
        x_train, y_train = self.train_data
        x_train = x_train.astype('float32')
        x_train /= 255
        y_train = to_categorical(y_train, self.n_classes)

        x_test, y_test = self.test_data
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = to_categorical(y_test, self.n_classes)
        
        self.train_data = (x_train, y_train)
        self.test_data = (x_test, y_test)

    def _model_fit(self):
        self.model.fit(x=self.train_data[0],
                       y=self.train_data[1],
                       epochs=self.epochs,
                       verbose=2)

    def _model_evaluate(self):
        [_, validate_acc] = self.model.evaluate(self.test_data[0], self.test_data[1])
        return validate_acc

train_data, test_data = cifar10.load_data()    
problem = CustomizedNSGANET(input_shape=(32, 32, 3),
                              n_classes=10,
                              data=(train_data, test_data, None),
                              activation_last='softmax',
                              optimizer='sgd',
                              loss='categorical_crossentropy',
                              stages=(6, 6),
                              activation='relu',
                              epochs=5)


termination = MaxGenTermination(200)

crossover = cx.MBUniformCrossover(prob=0.9)
mutation = mut.BitFlipMutation(prob=0.02)

algorithm = mo.NSGAII(pop_size=40,
                      crossover=crossover,
                      elitist_archive=2,
                      mutation=mutation)
algorithm.model = problem.problem_model

result = optimize(problem, 
                  algorithm, 
                  termination=termination, 
                  verbose=True,
                  log=True, 
                  save=True,
                  save_history=True, 
                  seed=1, 
                  display=display,
                  log_saver=log_saver)