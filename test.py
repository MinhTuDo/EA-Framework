import numpy as np
np.random.seed(1)

genome = np.random.randint(low=0, high=2, size=(69,))
indices = np.arange(len(genome))
print(genome)

n_nodes = [6, 6, 6]
input_size = (32, 32, 3)

connections_length = [((n*(n-1)) // 2) + 3 for n in n_nodes]
list_connections, list_indices = [], []
for i, (length, n_node) in enumerate(zip(connections_length, n_nodes)):
    phase = genome[i*length : (i+1)*length]
    phase_indices = indices[i*length : (i+1)* length]
    list_nodes, node_indices = [], []
    start = 0
    for i in range(1, n_node):
        end = start + i
        list_nodes += [phase[start : end].tolist()]
        node_indices += [phase_indices[start : end].tolist()]
        start = end
        
    list_nodes += [[phase[-3]], [int(''.join(str(bit) for bit in phase[-2:]), 2)]]
    node_indices += [*[[phase_indices[-3]], phase_indices[-2:].tolist()]]

    list_connections += [list_nodes]
    list_indices += [*node_indices]

n_bits = {'kernel_size': 2, 'pool_size': 1, 'channels': 2}
target_val = {'kernel_size': [3, 5, 7, 9],
              'pool_size': [1, 2],
              'channels': [16, 32, 64, 128]}
config = {}
bit_count = 0
for encode_name, n in n_bits.items():
    encode_indices = indices[::-1][bit_count : bit_count + (n*len(n_nodes))][::-1]
    list_indices += [encode_indices[i:i+n].tolist() for i in range(0, len(encode_indices), n)]

    encode_bits = genome[::-1][bit_count : bit_count + (n*len(n_nodes))][::-1]
    encode_val = [int(''.join(str(bit) for bit in encode_bits[i:i+n]), 2) for i in range(0, len(encode_bits), n)]
    target = np.array(target_val[encode_name])
    encode_val = target[encode_val].tolist()
    bit_count += n * len(n_nodes)
    config[encode_name] = encode_val

channels = config['channels']
new_channels = [None] * len(channels)
for i, channel in enumerate(channels):
    new_channels[i] = [channels[i-1] if i != 0 else input_size[2], channel]


print(config)

# print('connection')
# print(list_connections)
# print('model')
# print(list_indices)