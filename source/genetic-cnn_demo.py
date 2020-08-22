from model import Display, LogSaver
from factory import GAFactory
from optimize import optimize
from utils import GifSaver
from keras.utils import to_categorical
from keras.datasets import cifar10

import numpy as np

class MySaver(LogSaver):
    def _do(self, algorithm):
        self.add_attributes('n_gens', algorithm.n_gens) 
        self.add_attributes('n_evals', algorithm.n_evals) 
        self.add_attributes('F', algorithm.fitness_opt)

class MyDisplay(Display):
    def _do(self, algorithm):
        self.display_top = -1
        self.add_attributes('n_gens', algorithm.n_gens)
        self.add_attributes('n_evals', algorithm.n_evals)
        self.add_attributes('min', algorithm.fitness_pop.std(), width=5)
        self.add_attributes('mean', algorithm.fitness_pop.mean(), width=5)
        self.add_attributes('F', algorithm.fitness_opt)
        

display = MyDisplay()
log_saver = MySaver()
factory = GAFactory()

GeneticCNN = factory.get_problem('GeneticCNN')
class CustomizeGeneticCNN(GeneticCNN):
    def _preprocess_input(self):
        x_train, y_train = self.train_data
        x_train = x_train.astype('float32')
        x_train /= 255
        y_train = to_categorical(y_train, self.classes)

        x_test, y_test = self.test_data
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = to_categorical(y_test, self.classes)
        
        self.train_data = (x_train, y_train)
        self.test_data = (x_test, y_test)

    def _model_fit(self):
        self.model.fit(x=self.train_data[0],
                       y=self.train_data[1],
                       epochs=1)

    def _model_evaluate(self):
        self.model.evaluate(self.test_data[0], self.test_data[1])

train_data, test_data = cifar10.load_data()    
problem = CustomizeGeneticCNN(input_shape=(32, 32, 3),
                              classes=10,
                              data=(train_data, test_data, None),
                              activation_last='softmax',
                              optimizer='sgd',
                              loss='categorical_crossentropy',
                              stages=(2, 5),
                              fc=[120, 84],
                              activation='tanh')
termination = factory.get_termination('MaxGenTermination')(2)

crossover = factory.get_crossover('UniformCrossover')()

algorithm = factory.get_algorithm('SGA')(pop_size=4)

result = optimize(problem, 
                  algorithm, 
                  termination=termination, 
                  verbose=True,
                  log=False, 
                  save_history=True, 
                  seed=1, 
                  display=display,
                  log_saver=log_saver)


