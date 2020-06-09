import argparse

from .config import get_config
from .parameters import *
from .simulation import SetupSim


def main(config, n=1):
    for _ in range(n):
        simulation = SetupSim(config).setup()
        simulation.prepare()
        simulation.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config", type=str,
                        help="config file")
    parser.add_argument("-n", dest="n", type=int,
                        help="number of sim")
    args = parser.parse_args()

    main(get_config(args.config), args.n)