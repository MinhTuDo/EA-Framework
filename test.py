import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
from model import Display, LogSaver
import time

np.set_printoptions(suppress=True)

filename = './logs/obj_20201010-0726_gen-17__NSGAII-NSGANet_.pickle'
last_gen = None
with open(filename, 'rb') as handle:
    last_gen = pickle.load(handle)

filename = './logs/obj_20200930-2342_gen-1__NSGAII-NSGANet_.pickle'
init_gen = None
with open(filename, 'rb') as handle:
    init_gen = pickle.load(handle)

# print(gen17.ranks_F[0])
# print(gen17.ranks[0][7])

plt.plot(last_gen.F_pop[:, 1], last_gen.F_pop[:, 0], 'b.', label='Last Generation')
plt.plot(init_gen.F_pop[:, 1], init_gen.F_pop[:, 0], 'r.', label='Initialization')

last_sorted_idx = np.argsort(last_gen.ranks_F[0][:, 1])
init_sorted_idx = np.argsort(init_gen.ranks_F[0][:, 1])

plt.plot(last_gen.ranks_F[0][last_sorted_idx][:, 1], last_gen.ranks_F[0][last_sorted_idx][:, 0], color='blue', alpha=.5)
plt.plot(init_gen.ranks_F[0][init_sorted_idx][:, 1], init_gen.ranks_F[0][init_sorted_idx][:, 0], color='red', alpha=.5)

# plt.plot(init_gen.ranks_F[0][:, 1], init_gen.ranks_F[0][:, 0], 'rv', alpha=.5)
# plt.plot(last_gen.ranks_F[0][:, 1], last_gen.ranks_F[0][:, 0], 'b^', alpha=.5)
# plt.xlim([0, 3000])
# plt.xscale('log')
plt.xlabel('flops (Millions)')
plt.ylabel('error rate (%)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--')
plt.show()