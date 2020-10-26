from agents import *
import json
from utils.flops_benchmark import add_flops_counting_methods
import torch
import numpy as np  
import time
from torch.nn import Linear

def print_info(self):
    n_params = sum([np.prod(p.size()) for p in self.parameters]) / 1e6

    self.model = add_flops_counting_methods(self.model)
    self.model.eval()
    self.model.start_flops_count()
    random_data = torch.randn(1, *self.data_info['input_size'])
    self.model(torch.autograd.Variable(random_data).to(self.device))
    n_flops = (self.model.compute_average_flops_cost() / 1e6).round(4)

    print('{} Million of parameters | {} MFLOPS'.format(n_params, n_flops))

config = None
with open('./configs/transfer_learning.json') as  json_file:
    config = json.load(json_file)
agent_constructor = globals()[config['agent']]

agent = agent_constructor(**config, callback=print_info)

# model = agent.model
# for param in model.parameters():
#     param.requires_grad = False
# in_features = model.linear.in_features
# model.linear = Linear(in_features, 100)
# model.to(agent.device)

start = time.time()
agent.run()
agent.finalize()

end = time.time() - start

print('Elapsed time: {}'.format(end))