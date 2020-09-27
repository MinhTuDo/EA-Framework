# import numpy as np
# from keras.models import Model
# from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Dropout, MaxPool2D, AveragePooling2D
# from abc import abstractmethod
# from .so_problem import SingleObjectiveProblem


# class GeneticCNN(SingleObjectiveProblem):
#     def __init__(self,
#                  input_shape=None,
#                  n_classes=None,
#                  data=(None, None, None),
#                  activation_last='softmax',
#                  optimizer='adam',
#                  loss='categorical_crossentropy',
#                  stages=(3, 5),
#                  Pool=MaxPool2D,
#                  pool_size=(1, 1),
#                  init_filters=8,
#                  kernel_size=(5, 5),
#                  strides=(1, 1),
#                  fc=[300, 300],
#                  epochs=1,
#                  dropout=0.5,
#                  activation='relu',
#                  **kwargs):
#         super().__init__(n_params=sum(list(map(self.__calc_L, stages))),
#                          param_type = np.int,
#                          n_constraints=0,
#                          multi_dims=True)
#         xl = np.ones((self.n_params,)) * 0
#         xu = np.ones((self.n_params,)) * 1
#         self.domain = (xl, xu)
        
#         self.stages = stages
#         self.input_shape = input_shape
#         self.n_classes = n_classes
#         self.activation = activation
#         self.activation_last = activation_last
#         self.Pool = Pool
#         self.pool_size = pool_size
#         self.init_filters = init_filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.fc = fc
#         self.dropout = dropout
#         self.model = None
#         self.loss = loss
#         self.optimizer = optimizer

#         (self.train_data, self.test_data, self.validation_data) = data

#         self._optimum = max
#         self._argopt = np.argmax
#         self._preprocess_input()


#     def create_model(self, binary_string):
#         X_input = Input(self.input_shape)
#         X = X_input
        
#         for idx, K in enumerate(self.stages):
#             default_input = self.__conv(X, self.init_filters * (idx+1))
#             start_idx = 0 if idx == 0 else self.__calc_L(self.stages[idx-1])
#             end_idx = start_idx + self.__calc_L(K)
#             code = binary_string[start_idx : end_idx]
#             X = self.__connect_nodes(default_input=default_input, 
#                                      code=code, 
#                                      n_nodes=K, 
#                                      n_filters=self.init_filters * (idx+1))
#             X = self.Pool(pool_size=self.pool_size,
#                           strides=self.strides,
#                           padding='valid')(X)

#         X = Flatten()(X)
#         for n_units in self.fc:
#             X = Dense(n_units, activation=self.activation)(X)
#         X = Dropout(self.dropout)(X)
#         X = Dense(self.n_classes, activation=self.activation_last)(X)
        
#         model = Model(inputs=X_input, outputs=X)
#         model.compile(loss=self.loss,
#                       optimizer=self.optimizer,
#                       metrics=['accuracy'])

#         return model

#     ## Overide Methods ##
#     def _f(self, X):
#         self.model = self.create_model(X)
#         self._model_fit()
#         return self._model_evaluate()
    
#     def _sol_compare(self, y1, y2):
#         return y1 >= y2

#     ## Abstract Methods ##
#     @abstractmethod
#     def _preprocess_input(self):
#         pass
#     @abstractmethod
#     def _model_fit(self):
#         pass
#     @abstractmethod
#     def _model_evaluate(self):
#         pass


#     def __calc_L(self, k):
#         l = (k * (k - 1)) // 2
#         return l

#     def __conv(self, X, n_filters):
#         node = X
#         node = Conv2D(filters=n_filters,
#                       kernel_size=self.kernel_size,
#                       strides=self.strides,
#                       padding='same')(node)
#         node = BatchNormalization(axis=3)(node)
#         node = Activation(self.activation)(node)
#         return node


#     def __connect_nodes(self, 
#                         default_input, 
#                         code, 
#                         n_nodes, 
#                         n_filters):
#         if code.sum() == 0:
#             return self.__conv(default_input, n_filters)

#         nodes = np.arange(2, n_nodes+1)
#         end_indices = np.array(list(map(self.__calc_L, nodes)))
#         start_indices = end_indices - (nodes-1)
#         n_connections = [code[start_idx:end_idx] for start_idx, end_idx in zip(start_indices, end_indices)]
#         nodes = [None] * n_nodes
#         nodes[0] = self.__conv(default_input, n_filters)

#         for i, connection_i in enumerate(n_connections):
#             output_node = None
#             if connection_i.sum() == 0:
#                 is_isolated = True
#                 for j in range(i+1, len(n_connections)):
#                     if n_connections[j][i+1] == 1:
#                         is_isolated = False
#                         break
                        
#                 if not is_isolated:
#                     output_node = default_input
#                     nodes[i+1] = self.__conv(output_node, n_filters)
#             else:
#                 connected_nodes = np.where(connection_i == 1)[0]
#                 for node_i in connected_nodes:
#                     if output_node is None:
#                         output_node = nodes[node_i]
#                     else:
#                         output_node = Add()([output_node, nodes[node_i]])
#                 nodes[i+1] = self.__conv(output_node, n_filters)
                

#         for node_i in reversed(nodes):
#             if node_i is not None:
#                 return self.__conv(node_i, n_filters)

#         return None


    