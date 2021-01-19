import argparse

from gmplabtools.shared.config import get_config
from gmplabtools.simulations.simulation import SetupSim


def main(config):
    simulations = SetupSim(config).setup()
    for sim in simulations:
        sim.prepare().run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config", type=str,
                        help="config file")
    args = parser.parse_args()
    main(get_config(args.config))
