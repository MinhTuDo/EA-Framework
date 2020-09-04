import pickle5 as pickle

filename = './log/obj_20200904-1344.pickle'
obj = None
with open(filename, 'rb') as handle:
    obj = pickle.load(handle)

print('Debug')
obj