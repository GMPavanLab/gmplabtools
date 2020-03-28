import logging
import argparse

import numpy as np

from gmplabtools.pamm.lib.tools import GMMPredict


logging.getLogger(__name__).setLevel(level=logging.INFO)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate cluster probabilities using pamm output')
    parser.add_argument('-f', dest='filename', type=str,
                        help='filename to process')
    parser.add_argument('-t', dest='traj_pca', type=str,
                        help='pca file predict cluster probability')

    args = parser.parse_args()

    gmm_predict = GMMPredict.read_clusters(args.filename)
    x = np.loadtxt(args.traj_pca)

    predictions = gmm_predict.predict_proba(x)
    cluster = np.argmax(predictions, axis=1) + 1

    results = np.hstack((cluster.reshape((-1, 1)), predictions))

    index = args.filename.find('.pamm')
    output_filename = args.filename[:index] + '_results.txt'
    np.savetxt(output_filename, results, fmt='%1d' + predictions.shape[1]*' %1.6e', header='cluster_n, p_k=1,...,D')