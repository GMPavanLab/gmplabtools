import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from gmplabtools.shared.config import get_config
from gmplabtools.analysis import DataSampler, calculate_adjacency, adjancency_dendrogram
from gmplabtools.pamm.pamm import Pamm


def main(config):

    x = np.loadtxt(config.trj_filename)

    if config.generate_grid:
        d = DataSampler(config.distance, norm=config.p)
        grid, indices = d.minmax_sample(x, config.size)
        np.savetxt("{}.grid".format(config.savegrid), indices + 1, fmt="%d")

    p = Pamm(config.pamm_input)
    print(p.command_parser)
    p.run()

    if 'bootstrap' in config.pamm_input:
        adjacency, mapping = calculate_adjacency(
            prob=p.p,
            clusters=p.cluster,
            bootstrap=p.bs
        )

        z = adjancency_dendrogram(adjacency)
        fig, ax = plt.subplots()
        _ = dendrogram(z, ax=ax, **config.dendrogram)['leaves']
        fig.savefig('clusters_dendrogram.png')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config", type=str,
                        help="config file")
    args = parser.parse_args()

    main(get_config(args.config, "cluster"))