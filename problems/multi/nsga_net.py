import numpy as np
from abc import abstractmethod
from .mo_problem import MultiObjectiveProblem
import torch
from graphs.models.evo_net import EvoNet
from utils.flops_benchmark import add_flops_counting_methods
from agents import *
import time

class NSGANet(MultiObjectiveProblem):
    def __init__(self, 
                 n_bits, 
                 n_nodes, 
                 input_size,
                 output_size,
                 arch_config, 
                 predictor_config,
                 **kwargs):

        super().__init__(param_type=np.int,
                         n_constraints=0,
                         n_obj=2)
        n_encode_bits = n_bits
        max_node_per_phase = max(n_nodes)
        connection_bits = (max_node_per_phase * (max_node_per_phase-1)) // 2 + 2 + 1    # plus one residual bit & two node type bits
        self.n_params = (sum(n_encode_bits.values()) + connection_bits) * len(n_nodes)
        xl = np.zeros((self.n_params,))
        xu = np.ones((self.n_params,))
        self.domain = (xl, xu)

        agent_constructor = globals()[arch_config['agent']]
        self.agent = agent_constructor(arch_config)
        agent_constructor = globals()[predictor_config['agent']]
        self.predictor = agent_constructor(predictor_config)

        self.input_size = input_size
        self.output_size = output_size



    ## Overide Methods ##
    def _is_dominated(self, y1, y2):
        return (y1[0] <= y2[0] and y1[1] <= y2[1]) and \
               (y1[0] < y2[0] or y1[1] < y2[1])
    
    def _f(self, X):
        device = self.agent.device
    
        model = self.agent(genome=X, 
                           input_size=self.input_size, 
                           output_size=self.output_size).model

        n_params = self.__calc_trainable_params(self.agent.parameters)

        self.agent.to(device)
        _, train_error = self.agent.train_one_epoch()
        valid_error, infer_time = self.infer()

        n_flops = self.__calc_flops(model, device)

        _input = torch.tensor([train_error, valid_error, infer_time, n_params, n_flops], dtype=torch.float)
        predicted_valid_error = self.predictor.model(_input)
        self.agent.train() if predicted_valid_error < 20 else ...   # threshold

        valid_error, infer_time = self.infer()

        target = torch.tensor(valid_error, dtype=torch.float)
        self.predictor.model.train()
        self.predictor.feed_forward(_input, target)

        return valid_error, n_flops

    def infer(self):
        infer_time = time.time()
        _, valid_error = self.agent.validate()
        infer_time = time.time() - infer_time
        return valid_error, infer_time

    def __calc_flops(self, model, device):
        model = add_flops_counting_methods(model)
        model.eval()
        model.start_flops_count()
        random_data = torch.randn(1, *self.config['input_size'])
        model(torch.autograd.Variable(random_data).to(device))
        n_flops = (model.compute_average_flops_cost() / 1e6).round(4)
        return n_flops

    def __calc_trainable_params(self, params):
        return (p.size().prod() for p in params).sum() / 1e6
        
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


        
        