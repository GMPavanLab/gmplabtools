import argparse

import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP
import matplotlib.pyplot as plt

from gmplabtools.shared.config import get_config, import_external


def read_traj(filename, index=":", start=None, end=None, stride=None):
    if all([start, end, stride]):
        index = "{}:{}:{}".format(start, end, stride),
    return read(filename, index=index, format="xyz")


def plot(x, fname):
    plt.figure(figsize=(11, 8), dpi=80)
    plt.ylabel('2nd coord')
    plt.xlabel('1st coord')
    plt.scatter(x[:, 0], x[:, 1], c=x[:, 2], cmap='inferno')
    plt.colorbar()
    plt.savefig(fname)
    plt.close()


def main(config):

    n_traj = len(config.trajectories)
    traj = {name: read_traj(traj) for name, traj in config.transform["trajectories"].items()}

    all_traj = sum(traj.values(), [])

    # info on how it works (and installation):
    # https://singroup.github.io/dscribe/tutorials/soap.html
    soapDSC = SOAP(**config.soap_param)

    soap = {name: soapDSC.create(k) for name, k in traj.items()}
    all_soap = soapDSC.create(all_traj)

    print("DSCRIBE descriptor shapes:")
    msg = ", ".join(
        ["{}: {}".format(name, k.shape[0]) for name, k in soap.items()]
    )
    print(msg)

    transformer = import_external(config.transform["transformer"]["name"])
    params = config.transform["transformer"]["params"]
    transformer = transformer(**params)

    transformer = transformer.fit(all_soap)

    # calculate variance ratios on the merged data
    if "PCA" in str(config.transform["transformer"]["name"]):
        variance = transformer.explained_variance_ratio_
        var = np.cumsum(np.round(variance, decimals=3) * 100)
        print("PCA (dim={}) variance: {}".format(
            str(transformer.named_steps['pca'].n_components_),
            var)
        )

    transformed = {name: transformer.transform(k) for name, k in soap.items()}
    red_dim = transformer.transform(all_soap)

    np.savetxt("all_system_transformed.txt", red_dim)

    for k, x in transformed.items():
        np.savetxt("{}_soap.txt".format(k), x)

    sample_length = min(2000, x.shape[0])

    if config.plot:
        for k, x in transformed.items():
            np.random.shuffle(x)
            plot(x[:sample_length, :], "scatter_plot_{}.png".format(k))

    np.random.shuffle(all_pca)
    plot(all_pca[:sample_length * n_traj, :], "scatter_plot_all.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config", type=str,
                        help="config file")
    args = parser.parse_args()

    main(get_config(args.config))
