from agents import *
import json
from utils.flops_benchmark import add_flops_counting_methods
import torch
import numpy as np  
import time

def print_info(self):
    n_params = (np.sum(np.prod(v.size()) for v in filter(lambda p: p.requires_grad, self.model.parameters())) / 1e6)

    self.model = add_flops_counting_methods(self.model)
    self.model.eval()
    self.model.start_flops_count()
    random_data = torch.randn(1, *self.data_info['input_size'])
    self.model(torch.autograd.Variable(random_data).to(self.device))
    n_flops = np.round(self.model.compute_average_flops_cost() / 1e6, 4)

    print('{} Million of parameters | {} MFLOPS'.format(n_params, n_flops))

config = None
with open('./configs/train_arch.json') as  json_file:
    config = json.load(json_file)
agent_constructor = globals()[config['agent']]

agent = agent_constructor(**config, callback=print_info)

start = time.time()
agent.run()
agent.finalize()

end = time.time() - start

print('Elapsed time: {}'.format(end))