import numpy as np
from abc import abstractmethod
from .mo_problem import MultiObjectiveProblem
import torch
from graphs.models.evo_net import EvoNet
from utils.flops_benchmark import add_flops_counting_methods
from agents.dl_agent import DeepLearningAgent
import time

class NSGANet(MultiObjectiveProblem):
    def __init__(self, 
                 arch_config, 
                 predictor_config,
                 **kwargs):

        super().__init__(param_type=np.int,
                         n_constraints=0,
                         n_obj=2)
        n_encode_bits = arch_config['model_args']['n_bits']
        n_nodes = arch_config['model_args']['n_nodes']
        max_node_per_phase = max(n_nodes)
        connection_bits = (max_node_per_phase * (max_node_per_phase-1)) // 2 + 2 + 1    # plus one residual bit & two node type bits
        self.n_params = ((sum(n_encode_bits.values()) + connection_bits) * len(n_nodes)) - 1 # subtract kernel size bit encoding for last phase
        xl = np.zeros((self.n_params,))
        xu = np.ones((self.n_params,))
        self.domain = (xl, xu)
        random_genome = np.zeros((self.n_params,), dtype=self.param_type)
        _, self.problem_model = EvoNet.setup_model_args(genome=random_genome, **arch_config['model_args'])

        self.arch_config = arch_config

        # self.predictor = DeepLearningAgent(predictor_config)
        # self.predictor.model.train()
        # setattr(self.predictor, 'sample_count', 1)

        self.hash_dict = {}
        self.input_size = self.arch_config['model_args']['input_size']


    ## Overide Methods ##
    def _f(self, X):
        key = self.__get_key(X)
        if key in self.hash_dict:
            (valid_error, n_flops, _, _) = self.hash_dict[key]
            return valid_error, n_flops

        self.arch_config['model_args']['genome'] = X
        agent = DeepLearningAgent(**self.arch_config)

        n_params = self.__calc_trainable_params(agent.parameters)
        n_flops = self.__calc_flops(agent.model, agent.device, self.input_size)

        agent.train()
        valid_error, infer_time = self.__infer(agent)

        # _input = self.__create_sample(train_error, valid_error, infer_time, n_params, n_flops)
        # with torch.no_grad():
        #     pred_val_err = self.predictor.model(_input)
        # if pred_val_err < 20 or self.predictor.sample_count < 40:
        #     print('Retraining model...')
        #     agent.train()
        #     self.predictor.sample_count += 1
        #     valid_error, infer_time = self.__infer(agent)

        #     target = torch.tensor([valid_error], dtype=torch.float)
        #     self.predictor.feed_forward(_input, target)
        # else:
        #     print('Skip model!')

        self.hash_dict[key] = (valid_error, n_flops, n_params, infer_time)
        return valid_error, n_flops

    @staticmethod
    def _is_dominated(y1, y2):
        return (y1[0] <= y2[0] and y1[1] <= y2[1]) and \
               (y1[0] < y2[0] or y1[1] < y2[1])

    @staticmethod
    def __calc_flops(model, device, input_size):
        model = add_flops_counting_methods(model)
        model.eval()
        model.start_flops_count()
        random_data = torch.randn(1, *input_size)
        model(torch.autograd.Variable(random_data).to(device))
        n_flops = (model.compute_average_flops_cost() / 1e6).round(4)
        return n_flops

    @staticmethod
    def __calc_trainable_params(params):
        return sum([np.prod(p.size()) for p in params]) / 1e6

    @staticmethod
    def __get_key(X):
        key = ''.join(str(bit) for bit in X)
        return key

    @staticmethod
    def __create_sample(train_error, 
                        valid_error, 
                        infer_time, 
                        n_params, 
                        n_flops):
        _input = torch.tensor([train_error, valid_error, infer_time, n_params, n_flops], dtype=torch.float)
        _input = (_input - _input.mean()) / _input.std()
        return _input

    @staticmethod
    def __infer(agent):
        infer_time = time.time()
        valid_error, _ = agent.validate()
        infer_time = time.time() - infer_time
        return valid_error, infer_time