import argparse
import json

import numpy as np

from gmplabtools.pamm.lib.dimensionality import DataSampler     
from gmplabtools.pamm.pamm_commander import PammCommander  

from config import get_config


def main(config):

    x = np.loadtxt(config.trj_filename)
    np.random.shuffle(x)

    if config.generate_grid:
        d = DataSampler(config.distance, norm=config.p)   
        grid, indices = d.minmax_sample(x, config.size)
        np.savetxt("{}.grid".format(config.savegrid), indices + 1, fmt="%d")

    p = PammCommander(config.pamm_input)
    print(p.command_parser)
    p.run()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config", type=str,
                        help="config file")
    args = parser.parse_args()

    main(get_config(args.config, "cluster"))