from agents import *
import json
import time
from utils.displayer import print_info
import click

@click.command()
@click.option('--config', required=True)
def cli(config):
    with open('./configs/{}'.format(config)) as  json_file:
        config = json.load(json_file)
    seeds = list(range(10))
    for seed in seeds:
        print('Run seed: {}'.format(seed))
        config['seed'] = seed
        agent_constructor = globals()[config['agent']]

        agent = agent_constructor(**config, callback=print_info)

        start = time.time()
        agent.run()
        agent.finalize()

        end = time.time() - start

        print('Elapsed time: {}'.format(end))

if __name__ == '__main__':
    cli()