import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
from model import Display, LogSaver
import time

np.set_printoptions(suppress=True)

filename = './logs/obj_20201019-2148_gen-25__NSGAII-NSGANet_.pickle'
last_gen = None
with open(filename, 'rb') as handle:
    last_gen = pickle.load(handle)

filename = './logs/obj_20201019-1010_gen-24__NSGAII-NSGANet_.pickle'
init_gen = None
with open(filename, 'rb') as handle:
    init_gen = pickle.load(handle)

print(last_gen.ranks_F[0][3])
print(last_gen.ranks[0][3])

plt.plot(last_gen.F_pop[:, 1], last_gen.F_pop[:, 0], 'b.', label='final pop')
plt.plot(init_gen.F_pop[:, 1], init_gen.F_pop[:, 0], 'r.', label='init pop')

last_sorted_idx = np.argsort(last_gen.ranks_F[0][:, 1])
init_sorted_idx = np.argsort(init_gen.ranks_F[0][:, 1])

plt.plot(last_gen.ranks_F[0][last_sorted_idx][:, 1], 
         last_gen.ranks_F[0][last_sorted_idx][:, 0], color='blue', alpha=.5, label='final front')
plt.plot(init_gen.ranks_F[0][init_sorted_idx][:, 1], 
         init_gen.ranks_F[0][init_sorted_idx][:, 0], color='red', alpha=.5, label='initial front')

# plt.plot(init_gen.ranks_F[0][:, 1], init_gen.ranks_F[0][:, 0], 'rv', alpha=.5)
# plt.plot(last_gen.ranks_F[0][:, 1], last_gen.ranks_F[0][:, 0], 'b^', alpha=.5)
plt.ylim([7, 11])
plt.xlim([100, 2000])
# plt.xscale('log')
plt.xlabel('flops (millions)')
plt.ylabel('error rate (%)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--')
plt.show()

# 0 0 1 0 0 0 0 0 1 0 0 0 1 1 1 0 0 0 0 0 1 1 1 0 0 1 0 1 0 0 1 1 0 1 0 0 0 0 0 0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 1
