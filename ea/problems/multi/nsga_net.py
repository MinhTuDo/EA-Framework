import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Dropout, MaxPool2D, AveragePooling2D
from abc import abstractmethod
from .mo_problem import MultiObjectiveProblem
from keras_flops import get_flops

class NSGANET(MultiObjectiveProblem):
    def __init__(self,
                 input_shape=None,
                 n_classes=None,
                 activation='relu',
                 activation_last='softmax',
                 optimizer='adam',
                 loss='categorical_crossentropy',
                 stages=(6, 6),
                 epochs=1,
                 dropout=0.5,
                 **kwargs):
        super().__init__(param_type=np.int,
                         n_constraints=0)
        