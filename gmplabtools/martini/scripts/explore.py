import argparse

from gmplabtools.shared.config import get_config
from gmplabtools.martini.simulation import SetupSim


def main(config, n=1):
    for _ in range(n):
        simulation = SetupSim(config).setup()
        simulation.prepare()
        simulation.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config", type=str,
                        help="config file")
    args = parser.parse_args()
    main(get_config(args.config), args.n)