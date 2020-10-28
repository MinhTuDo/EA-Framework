import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

filename = './logs/run_1_seed_1.pickle'
with open(filename, 'rb') as handle:
    gens = pickle.load(handle)

for i in range(1, len(gens)-1):
    mid_gen = gens[i]
    plt.scatter(mid_gen.F_pop[:, 1], mid_gen.F_pop[:, 0], color='green', marker='.')

init_gen = gens[0]
last_gen = gens[-1]
print(last_gen.ranks_F[0][0])
print(last_gen.ranks[0][0])

plt.scatter(init_gen.F_pop[:, 1], init_gen.F_pop[:, 0], color='#ff441f', marker='.')
plt.scatter(last_gen.F_pop[:, 1], last_gen.F_pop[:, 0], color='#6d1fff', marker='.')

last_sorted_idx = np.argsort(last_gen.ranks_F[0][:, 1])
init_sorted_idx = np.argsort(init_gen.ranks_F[0][:, 1])

plt.plot(last_gen.ranks_F[0][last_sorted_idx][:, 1], 
         last_gen.ranks_F[0][last_sorted_idx][:, 0], color='blue', label='Final front', linewidth=2)
plt.plot(init_gen.ranks_F[0][init_sorted_idx][:, 1], 
         init_gen.ranks_F[0][init_sorted_idx][:, 0], color='red', label='Initial front', linewidth=2)

plt.plot(init_gen.ranks_F[0][:, 1], init_gen.ranks_F[0][:, 0], 'ro')
plt.plot(last_gen.ranks_F[0][:, 1], last_gen.ranks_F[0][:, 0], 'b^')
plt.ylim(bottom=6)
plt.xlim(right=2000, left=-100)
# plt.xscale('log')
plt.xlabel('Flops (Millions)')
plt.ylabel('Error Rate (%)')
plt.title('Trade-off frontiers')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--')
# plt.savefig('./assets/nsga_net_fig2.png')
plt.show()

# 0 0 1 0 0 0 0 0 1 0 0 0 1 1 1 0 0 0 0 0 1 1 1 0 0 1 0 1 0 0 1 1 0 1 0 0 0 0 0 0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 1
