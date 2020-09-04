import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Dropout, MaxPool2D, AveragePooling2D, ZeroPadding2D
from abc import abstractmethod
from .mo_problem import MultiObjectiveProblem
import tensorflow as tf
import keras.backend as K
from keras_flops import get_flops
import keras

class NSGANET(MultiObjectiveProblem):
    def __init__(self,
                 input_shape=None,
                 data=(None, None, None),
                 n_classes=None,
                 activation='relu',
                 activation_last='softmax',
                 optimizer='sgd',
                 loss='categorical_crossentropy',
                 stages=(6, 6),
                 epochs=1,
                 dropout=0.5,
                 **kwargs):
        super().__init__(param_type=np.int,
                         n_constraints=0,
                         n_obj=2)
        self.n_params = sum(list(map(self.__calc_L, stages))) + 15
        xl = np.zeros((self.n_params,))
        xu = np.ones((self.n_params,))
        self.domain = (xl, xu)
        self.problem_model = [[0, 1, 2], 
                              [3, 4, 5, 6, 7, 8, 9],
                              [10, 11],
                              [12, 13],
                              [14],
                              list(range(15, self.n_params))]
        self.stages = stages
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.activation = activation
        self.activation_last = activation_last
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.model = None
        self.binary_string = None

        (self.train_data, self.test_data, self.validation_data) = data

        self.epochs = epochs

        self._preprocess_input()

    def create_model(self, binary_string):
        X_input = Input(self.input_shape)
        X = X_input

        self.binary_string = binary_string
        attributes = list(map(self.__decode, self.problem_model))
        kernel_size = attributes[0]
        n_filters = attributes[1]
        pool_size = attributes[2]
        pool_strides = attributes[3]
        Pool = MaxPool2D if attributes[4] == 1 else AveragePooling2D
        code = self.binary_string[self.problem_model[-1]]
        for s, K_s in enumerate(self.stages):
            default_input = self.__conv(X, n_filters, kernel_size)
            start_idx = 0 if s == 0 else self.__calc_L(self.stages[s-1])
            end_idx = start_idx + self.__calc_L(K_s)
            X = self.__connect_nodes(default_input=default_input,
                                     code=code,
                                     n_nodes=K_s,
                                     n_filters=n_filters,
                                     kernel_size=kernel_size)
            X = Pool(pool_size=pool_size,
                     strides=pool_strides,
                     padding='valid')(X)

        X = Flatten()(X)
        X = Dropout(self.dropout)(X)
        X = Dense(self.n_classes, activation=self.activation_last)(X)

        model = Model(inputs=X_input, outputs=X)

        flops = get_flops(model)

        steps = self.epochs
        lr = tf.keras.experimental.CosineDecay(initial_learning_rate=0.025, decay_steps=steps)
        opt = keras.optimizers.SGD(learning_rate=0.025)

        model.compile(loss=self.loss,
                      optimizer=opt,
                      metrics=['accuracy'])

        return model, flops
    ## Overide Methods ##
    def _is_dominated(self, y1, y2):
        return (y1[0] <= y2[0] and y1[1] <= y2[1]) and \
               (y1[0] < y2[0] or y1[1] < y2[1])
    
    def _f(self, X):
        self.model, flops = self.create_model(X)
        self._model_fit()
        error_rate = 1 - self._model_evaluate()
        flops = flops / 1e5
        return error_rate, flops

        tf.compat.v1.reset_default_graph()


        return flops.total_float_ops
    ## Abstract Methods ##
    @abstractmethod
    def _preprocess_input(self):
        pass
    @abstractmethod
    def _model_fit(self):
        pass
    @abstractmethod
    def _model_evaluate(self):
        pass

    ## Private Methods ##
    def __connect_nodes(self,
                        default_input,
                        code,
                        n_nodes,
                        n_filters,
                        kernel_size):

        if code.sum() == 0:
            return self.__conv(default_input, n_filters, kernel_size, strides)

        nodes = np.arange(2, n_nodes+1)
        end_indices = np.array(list(map(self.__calc_L, nodes)))
        start_indices = end_indices - (nodes-1)
        n_connections = [code[start_idx:end_idx] for start_idx, end_idx in zip(start_indices, end_indices)]
        nodes = [None] * n_nodes
        nodes[0] = self.__conv(default_input, n_filters, kernel_size)

        for i, connection_i in enumerate(n_connections):
            output_node = None
            if connection_i.sum() == 0:
                is_isolated = True
                for j in range(i+1, len(n_connections)):
                    if n_connections[j][i+1] == 1:
                        is_isolated = False
                        break
                        
                if not is_isolated:
                    output_node = default_input
                    nodes[i+1] = self.__conv(output_node, n_filters, kernel_size)
            else:
                connected_nodes = np.where(connection_i == 1)[0]
                for node_i in connected_nodes:
                    if output_node is None:
                        output_node = nodes[node_i]
                    else:
                        output_node = Add()([output_node, nodes[node_i]])
                nodes[i+1] = self.__conv(output_node, n_filters, kernel_size)

        for node_i in reversed(nodes):
            if node_i is not None:
                return self.__conv(node_i, n_filters, kernel_size)

        return None

    def __decode(self, _range):
        return int(''.join(str(bit) for bit in self.binary_string[_range]), base=2) + 1

    def __conv(self, X, n_filters, kernel_size):
        node = X
        node = Conv2D(filters=n_filters,
                      kernel_size=kernel_size,
                      strides=1,
                      padding='same')(node)
        node = BatchNormalization(axis=3)(node)
        node = Activation(self.activation)(node)
        return node

    def __calc_L(self, k):
        l = (k * (k - 1)) // 2
        return l


        
        