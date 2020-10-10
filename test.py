import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
from model import Display, LogSaver
import time

np.set_printoptions(suppress=True)

filename = './logs/obj_20201010-0726_gen-17__NSGAII-NSGANet_.pickle'
gen17 = None
with open(filename, 'rb') as handle:
    gen17 = pickle.load(handle)

filename = './logs/obj_20201007-0006_gen-10__NSGAII-NSGANet_.pickle'
gen10 = None
with open(filename, 'rb') as handle:
    gen10 = pickle.load(handle)

print(gen17.ranks_F[0])
print(gen17.ranks[0][7])

plt.plot(gen17.F_pop[:, 1], gen17.F_pop[:, 0], 'b.', label='gen-17', alpha=.5)
plt.plot(gen10.F_pop[:, 1], gen10.F_pop[:, 0], 'r.', label='gen-10', alpha=.5)
plt.xlabel('flops (MB)')
plt.ylabel('error rate (%)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--')
plt.show()