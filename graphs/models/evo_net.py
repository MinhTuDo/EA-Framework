import torch
from torch.nn import Module
from utils.decoder import VariableGenomeDecoder
import torch.nn as nn
import numpy as np

class EvoNet(Module):
    n_bits = {'kernel_size': 2, 'pool_size': 1, 'channels': 2}
    target_val = {'kernel_size': [3, 5, 7, 9],
                  'pool_size': [1, 2],
                  'channels': [16, 32, 64, 128]}
    def __init__(self, 
                 genome, 
                 input_size, 
                 output_size, 
                 n_nodes):
        
        super(EvoNet, self).__init__()

        genome_dict = {}
        bit_count = 0
        for encode_name, n in n_bits.items():
            n_repeats = len(n_nodes) if encode_name != 'pool_size' else len(n_nodes)-1
            encode_bits = genome[::-1][bit_count : bit_count+(n*n_repeats)][::-1]
            encode_val = [int(''.join(str(bit) for bit in encode_bits[i:i+n]), 2) for i in range(0, n_repeats, n)]
            target = np.array(target_val[encode_name])
            encode_val = target[encode_val]
            bit_count += n * n_repeats
            genome_dict[encode_name] = encode_val

        connections_length = [((n*(n-1)) // 2) + 3 for n in n_nodes]
        list_connections = []
        for i, (length, n_node) in enumerate(zip(connections_length, n_nodes)):
            phase = genome[i*length : (i+1)*length]
            list_nodes = []
            start = 0
            for i in range(1, n_node):
                end = start + i
                list_nodes += [phase[start : end]]
                start = end
                
            list_nodes += [[phase[-3]], [int(''.join(str(bit) for bit in phase[-2:]), 2)]]
            list_connections += [list_nodes]

        genome_dict['list_genomes'] = list_connections

        self.model = VariableGenomeDecoder(**genome_dict, repeats=None).get_model()

        out = self.model(torch.autograd.Variable(torch.zeros(1, *input_size)))
        shape = out.data.shape

        self.gap = nn.AvgPool2d(kernel_size=(shape[-2], shape[-1]), stride=1)

        shape = self.gap(out).data.shape

        self.linear = nn.Linear(shape[1]*shape[2]*shape[3], output_size)

        self.model.zero_grad()

    def forward(self, x):
        x = self.gap(self.model(x))
        x = x.flatten()
        return self.linear(x)
