import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

filename = './logs/run_1_seed_1.pickle'
gen = None
with open(filename, 'rb') as handle:
    gen = pickle.load(handle)

gen_12 = gen[23]

#gen_last = gen[-1]

filename = 'logs/obj_20210323-1403_gen-24__NSGAII-NSGANet_.pickle'
init_gen = None
with open(filename, 'rb') as handle:
    init_gen = pickle.load(handle)

print(gen_12.ranks_F[0][0])
print(gen_12.ranks[0][0])

plt.plot(gen_12.F_pop[:, 1], gen_12.F_pop[:, 0], 'b.', label='mb 24')
plt.plot(init_gen.F_pop[:, 1], init_gen.F_pop[:, 0], 'r.', label='ux 24')
# plt.plot(gen_last.F_pop[:, 1], gen_last.F_pop[:, 0], 'g.', label='mb last')
# last_sorted_idx = np.argsort(last_gen.ranks_F[0][:, 1])
# init_sorted_idx = np.argsort(init_gen.ranks_F[0][:, 1])

# plt.plot(last_gen.ranks_F[0][last_sorted_idx][:, 1], 
#          last_gen.ranks_F[0][last_sorted_idx][:, 0], color='blue', alpha=.5, label='final front')
# plt.plot(init_gen.ranks_F[0][init_sorted_idx][:, 1], 
#          init_gen.ranks_F[0][init_sorted_idx][:, 0], color='red', alpha=.5, label='initial front')

# plt.plot(init_gen.ranks_F[0][:, 1], init_gen.ranks_F[0][:, 0], 'rv', alpha=.5)
# plt.plot(last_gen.ranks_F[0][:, 1], last_gen.ranks_F[0][:, 0], 'b^', alpha=.5)
# plt.ylim([7, 11])
# plt.xlim([100, 3000])
# plt.xscale('log')
plt.xlabel('flops (millions)')
plt.ylabel('error rate (%)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--')
plt.show()
