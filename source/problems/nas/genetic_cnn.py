import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Dropout, MaxPool2D, AveragePooling2D
class GeneticCNN:
    def __init__(self,
                 encode,
                 stages=(3, 5),
                 pool=MaxPool2D,
                 pool_size=(1, 1),
                 init_filters=8,
                 kernel_size=(5, 5),
                 strides=(1, 1),
                 fc=[300, 300],
                 dropout=0.5,
                 activation='relu'
                 ):
        assert len(encode) == sum(list(map(self.__code_length_per_stage, stages)))
        self.param_type = np.int
        self.encode = encode
        self.stages = stages
        self.input_shape = input_shape
        self.classes = classes
        self.hardcode = hardcode
        self.activation = activation

    def create_model(self, 
                     input_shape, 
                     classes,
                     optimizer='adam'):
        X_input = Input(input_shape)
        X = X_input

        k_nodes = self.stages[0]
        end_idx = self.__code_length_per_stage(k_nodes)
        code = self.encode[:end_idx]

        pass

    def __code_length_per_stage(self, k):
        l = (k * (k - 1)) // 2

    def __conv_layer(self, X, f):
        node = X
        node = Conv2D(filters=f,
                      kernel_size=self.kernel_size,
                      strides=self.strides,
                      padding='same')(node)
        node = BatchNormalization(axis=3)(node)
        node = Activation(self.activation)(node)
        return node

    def __connect_nodes(self,
                        X, 
                        code,
                        k_nodes,
                        f):
        a0 = self.__conv_layer(X, f)
        nodes = { 'a1': self.__conv_layer(a0, f) }

        for node in range(2, k_nodes+1):
            nodes['a' + str(node)] = a0
            end_idx = self.__code_length_per_stage(node)
            start_idx = end_idx - (node-1)
            prev_nodes = code[start_idx : end_idx]
            connected_nodes = np.where(prev_nodes == 1)[0] + 1
            for prev_node in connected_nodes:
                if (prev_node == node-1):
                    pass


    