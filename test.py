import numpy as np
np.random.seed(1)

genome = np.random.randint(low=0, high=2, size=(66,))
print(genome)

n_bits = {'kernel_size': 2, 'pool_size': 1, 'channels': 2}
target_val = {'kernel_size': [3, 5, 7, 9],
              'pool_size': [1, 2],
              'channels': [16, 32, 64, 128]}
config = {}
n_nodes = [6, 6, 6]
bit_count = 0
for encode_name, n in n_bits.items():
    encode_bits = genome[::-1][bit_count : bit_count+(n*len(n_nodes))][::-1]
    encode_val = [int(''.join(str(bit) for bit in encode_bits[i:i+n]), 2) for i in range(0, len(encode_bits), n)]
    target = np.array(target_val[encode_name])
    encode_val = target[encode_val]
    bit_count += n * len(n_nodes)
    config[encode_name] = encode_val
# kernel_size_bits = genome[::-1][:len(n_nodes)*2][::-1]
# print(kernel_size_bits)
# kernel_sizes = [int(''.join(str(bit) for bit in kernel_size_bits[i:i+2]), 2) for i in range(0, len(kernel_size_bits), 2)]
# kernel_sizes = np.array(list(range(3, 11, 2)))[kernel_sizes].clip(3, 7)

# pool_size_bits = genome[::-1][len(n_nodes)*2 : (len(n_nodes)*2) + len(n_nodes)][::-1]
# print(pool_size_bits)


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

print(list_connections)
print(connections_length)