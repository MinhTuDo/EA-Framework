import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

run1 = './logs/run_1_seed_1.pickle'
with open(run1, 'rb') as handle:
    gens1 = pickle.load(handle)

run2 = './logs/run_2_seed_0.pickle'
with open(run2, 'rb') as handle:
    gens2 = pickle.load(handle)


last_gen1 = gens1[-1]
last_gen2 = gens2[-1]

plt.scatter(last_gen1.F_pop[:, 1], last_gen1.F_pop[:, 0], color='#ff441f', marker='.', alpha=.75)
plt.scatter(last_gen2.F_pop[:, 1], last_gen2.F_pop[:, 0], color='#6d1fff', marker='.', alpha=.75)

last_sorted_idx = np.argsort(last_gen2.ranks_F[0][:, 1])
init_sorted_idx = np.argsort(last_gen1.ranks_F[0][:, 1])


plt.plot(last_gen1.ranks_F[0][init_sorted_idx][:, 1], 
         last_gen1.ranks_F[0][init_sorted_idx][:, 0], color='red', label='Front run 1', linewidth=2, linestyle='--')
plt.plot(last_gen2.ranks_F[0][last_sorted_idx][:, 1], 
         last_gen2.ranks_F[0][last_sorted_idx][:, 0], color='blue', label='Front run 2', linewidth=2, linestyle='--')

plt.plot(last_gen1.ranks_F[0][:, 1], last_gen1.ranks_F[0][:, 0], 'ro')
plt.plot(last_gen2.ranks_F[0][:, 1], last_gen2.ranks_F[0][:, 0], 'b^')
plt.ylim(bottom=6)
plt.xlim(right=2000, left=-100)
# plt.xscale('log')
plt.xlabel('Floating-point operations (Millions)')
plt.ylabel('Error Rate (%)')
plt.title('Trade-off frontiers (Seed 0)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--')
plt.savefig('./assets/compare.pdf')
plt.show()

# 0 0 1 0 0 0 0 0 1 0 0 0 1 1 1 0 0 0 0 0 1 1 1 0 0 1 0 1 0 0 1 1 0 1 0 0 0 0 0 0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 1
