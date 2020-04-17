import argparse
import json

import numpy as np

from gmplabtools.pamm.lib.transition_rates import ClusterRates
from gmplabtools.pamm.lib.tools import GMMPredict

from config import get_config

    
def main(config):
    gmm = GMMPredict.read_clusters(config.pamm_output + ".pamm")

    print("There are {} clusters".format(np.unique(gmm.pk).shape[0]))

    for f in config.extrapolate_on_files:
        x = np.loadtxt(f)
        x_ = gmm.predict_proba(x)
        clusters = np.argmax(x_, axis=1).reshape((-1, 1))
        save = np.hstack((x_, clusters))
        np.savetxt(f[:-4] + "_result.txt", save)

        rates = ClusterRates(40, "proba").calculate_matrix(clusters)
        np.savetxt(rates + "_rates.txt", save)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config", type=str,
                        help="config file")
    args = parser.parse_args()

    main(get_config(args.config, "predict"))