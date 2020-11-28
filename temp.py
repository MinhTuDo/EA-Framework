from agents import *
import json
import time
from utils.displayer import print_info
import click

config = 'valid.json'

with open('./configs/{}'.format(config)) as  json_file:
        config = json.load(json_file)
agent_constructor = globals()[config['agent']]

agent = agent_constructor(**config, callback=print_info)

start = time.time()
agent.run()
agent.finalize()

end = time.time() - start

print('Elapsed time: {}'.format(end))