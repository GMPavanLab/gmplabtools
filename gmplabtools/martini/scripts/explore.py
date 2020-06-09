import argparse

from .config import get_config
from .parameters import *
from .simulation import SetupSim


def main(config):

    for simulation in SetupSim(config):
        simulation.setup()
        simulation.prepare()
        simulation.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config", type=str,
                        help="config file")
    args = parser.parse_args()

    main(get_config(args.config))