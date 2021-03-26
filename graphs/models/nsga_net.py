import torch
from torch.nn import Module
from utils.macro_models import get_decoder
import torch.nn as nn

class NSGANetwork(Module):
    def __init__(self, 
                 genome, 
                 input_size, 
                 output_size, 
                 channels,
                 **kwargs):
        
        super(NSGANetwork, self).__init__()
        self.__name__ = 'NSGANetwork'

        # genome = genome if type(genome) == type(np.ndarray) else np.array([bit for bit in genome.replace(' ', '')], dtype=np.int)

        # genome_dict, list_indices = self.setup_model_args(n_nodes, genome, n_bits, target_val, input_size)

        self.model = get_decoder('dense', genome, channels).get_model()

        out = None
        with torch.no_grad():
            out = self.model(torch.autograd.Variable(torch.zeros(1, *(input_size))))
        shape = out.data.shape

        self.gap = nn.AvgPool2d(kernel_size=(shape[-2], shape[-1]), stride=1)

        shape = self.gap(out).data.shape

        self.linear = nn.Linear(shape[1]*shape[2]*shape[3], output_size)

    def forward(self, x):
        x = self.gap(self.model(x))
        x = x.view(x.size(0), -1)
        return self.linear(x)
