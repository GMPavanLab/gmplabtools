import argparse

import numpy as np

from gmplabtools.shared.config import get_config
from gmplabtools.analysis import ClusterRates
from gmplabtools.pamm import PammGMM


def main(config):
    gmm = PammGMM.read_clusters(config.cluster["o"] + ".pamm")

    print("There are {} clusters".format(np.unique(gmm.pk).shape[0]))

    for k, f in config.extrapolate_on_files.items():
        x = np.loadtxt(f)
        x_ = gmm.predict_proba(x)
        clusters = np.argmax(x_, axis=1).reshape((-1, 1))
        save = np.hstack((x_, clusters))
        np.savetxt(f[:-4] + "_result.txt", save)

        rates = ClusterRates(config.size[k], "label").calculate_matrix(save[:, -1])
        np.savetxt(f[:-4] + "_rates.txt", rates)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config", type=str,
                        help="config file")
    args = parser.parse_args()

    main(get_config(args.config))
