import os
import json
import re
from pickle5 import pickle
from graphs.models.evo_net import EvoNet
from utils.flops_benchmark import add_flops_counting_methods
import torch
import numpy as np
import matplotlib.pyplot as plt

def calc_flops(model, input_size):
    model = add_flops_counting_methods(model)
    model.eval()
    model.start_flops_count()
    random_data = torch.randn(1, *input_size)
    model(torch.autograd.Variable(random_data).to('cpu'))
    n_flops = np.round(model.compute_average_flops_cost() / 1e6, 4)
    return n_flops


path = './logs/'
gens = []
for filename in os.listdir(path=path):
    if re.match('.*-NSGANet_.pickle', filename):
        with open(os.path.join(path, filename), 'rb') as handle:
            gen = pickle.load(handle)
        gens += [gen]

with open('./configs/train_arch.json') as json_file:
    config = json.load(json_file)

model_args = config['model_args']
input_size = model_args['input_size']
del model_args['genome']

for idx, gen in enumerate(gens):
    print('processing gen {} ...'.format(idx+1))
    nets = list(map(lambda g : EvoNet(g, **model_args), gen.pop))
    flops = np.array(list(map(lambda n : calc_flops(n, input_size), nets)))
    gen.F_pop[:, 1] = flops
    gen.non_dominated_rank()
    print('Done')

sorted_gens = sorted(gens, key=lambda g : g.n_gens)
filename = './logs/run_2_seed_0.pickle'
with open(filename, 'wb') as handle:
    pickle.dump(sorted_gens, handle, protocol=pickle.HIGHEST_PROTOCOL)