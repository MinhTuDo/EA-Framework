import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Dropout, MaxPool2D, AveragePooling2D
from abc import abstractmethod
from model.problem import Problem

class GeneticCNN(Problem):
    def __init__(self,
                 input_shape=None,
                 classes=None,
                 data=(None, None, None),
                 activation_last='softmax',
                 optimizer='adam',
                 loss='categorical_crossentropy',
                 stages=(3, 5),
                 Pool=MaxPool2D,
                 pool_size=(1, 1),
                 init_filters=8,
                 kernel_size=(5, 5),
                 strides=(1, 1),
                 fc=[300, 300],
                 epochs=1,
                 dropout=0.5,
                 activation='relu',
                 **kwargs):
        super().__init__(n_params=sum(list(map(self.__code_length_per_stage, stages))),
                         param_type = np.int,
                         n_obj=-1,
                         n_constraints=0,
                         domain=(0, 1),
                         multi_dims=True)
        
        self.stages = stages
        self.input_shape = input_shape
        self.classes = classes
        self.activation = activation
        self.activation_last = activation_last
        self.Pool = Pool
        self.pool_size = pool_size
        self.init_filters = init_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.fc = fc
        self.dropout = dropout
        self.model = None
        self.loss = loss
        self.optimizer = optimizer

        self.train_data = data[0]
        self.test_data = data[1]
        self.validation_data = data[2]

        self._optimum = max
        self._argopt = np.argmax
        self._preprocess_input()


    def create_model(self, binary_string):
        X_input = Input(self.input_shape)
        X = X_input
        
        for idx, K in enumerate(self.stages):
            start_idx = 0 if idx == 0 else self.__code_length_per_stage(self.stages[idx-1])
            end_idx = start_idx + self.__code_length_per_stage(K)
            code = binary_string[start_idx : end_idx]
            X = self.__connect_nodes(X, 
                                     code, 
                                     K, 
                                     self.init_filters * (idx+1))
            X = self.Pool(pool_size=self.pool_size,
                          strides=self.strides,
                          padding='valid')(X)

        X = Flatten()(X)
        for n_units in self.fc:
            X = Dense(n_units, activation=self.activation)(X)
        X = Dropout(self.dropout)(X)
        X = Dense(self.classes, activation=self.activation_last)(X)
        
        model = Model(inputs=X_input, outputs=X)
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])

        return model

    ## Overide Methods ##
    def _function(self, X):
        self.model = self.create_model(X)
        self._model_fit()
        return self._model_evaluate()
    
    def _f_comparer(self, y1, y2):
        return y1 >= y2

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


    def __code_length_per_stage(self, k):
        l = (k * (k - 1)) // 2
        return l

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
                        K,
                        f):
        a0 = self.__conv_layer(X, f)
        nodes = { 'a1': self.__conv_layer(a0, f) }

        for node in range(2, K+1):
            nodes['a'+str(node)] = a0
            end_idx = self.__code_length_per_stage(node)
            start_idx = end_idx - (node-1)
            prev_nodes = code[start_idx : end_idx]
            connected_nodes = np.where(prev_nodes == 1)[0] + 1
            for prev_node in connected_nodes:
                if (prev_node == node-1):
                    nodes['a'+str(node)] = self.__conv_layer(nodes['a'+str(node-1)], f)
                else:
                    nodes['a'+str(node)] = Add()([nodes['a'+str(node)],
                                                  nodes['a'+str(prev_node)]])
            
        node_last = self.__conv_layer(nodes['a'+str(K)], f)   
        return node_last



    