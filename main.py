from agents import *
import json

config = None
with open('./configs/evo_net.json') as  json_file:
    config = json.load(json_file)
agent_constructor = globals()[config['agent']]

agent = agent_constructor(config)
agent.run()
agent.finalize()