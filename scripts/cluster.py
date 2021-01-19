import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from gmplabtools.shared import get_config
from gmplabtools.analysis import DataSampler, calculate_adjacency, adjancency_dendrogram
from gmplabtools.pamm import Pamm


def main(config):

    x = np.loadtxt("all_system_transformed")

    if config.cluster["generate_grid"]:
        d = DataSampler(config.distance, norm=config.p)
        grid, indices = d.minmax_sample(x, config.size)
        np.savetxt("{}.grid".format(config.savegrid), indices + 1, fmt="%d")

    p = Pamm(config.cluster["pamm_input"])
    p.run()

    if "bootstrap" in config.cluster["pamm_input"]:
        adjacency, mapping = calculate_adjacency(
            prob=p.p,
            clusters=p.cluster,
            bootstrap=p.bs
        )

        z = adjancency_dendrogram(adjacency)
        fig, ax = plt.subplots()
        _ = dendrogram(z, ax=ax, **config.cluster["dendrogram"])["leaves"]
        fig.savefig("clusters_dendrogram.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config", type=str,
                        help="config file")
    args = parser.parse_args()

    main(get_config(args.config))
