from agents import *
import json
import time
import click

@click.command()
@click.option('--config', required=True)
def cli(config):
    with open('./configs/{}'.format(config)) as  json_file:
        config = json.load(json_file)
    agent_constructor = globals()[config['agent']]

    agent = agent_constructor(**config)
    start = time.time()

    agent.run()
    agent.finalize()
    agent.problem._plot(agent.algorithm.result, plot_3D=False)
    end = time.time() - start

    print('Elapsed time: {}'.format(end))

if __name__ == '__main__':
    cli(['--config', 'fake_data.json'])