from agents import *
import json
import time

config = None
with open('./configs/evo_net.json') as  json_file:
    config = json.load(json_file)
agent_constructor = globals()[config['agent']]

agent = agent_constructor(**config)
start = time.time()

agent.run()
agent.finalize()
end = time.time() - start

print('Elapsed time: {}'.format(end))