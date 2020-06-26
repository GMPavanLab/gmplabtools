import argparse

import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from gmplabtools.shared.config import get_config


def read_traj(filename, index=":", start=None, end=None, stride=None):
    if all([start, end, stride]):
        index = "{}:{}:{}".format(start, end, stride),
    return read(filename, index=index, format="xyz")


def plot(pca, fname):
    plt.figure(figsize=(11, 8), dpi=80)
    plt.ylabel('2nd Coord')
    plt.xlabel('1st Coord')
    plt.title('2D PS - C1 fiber')
    plt.style.context('seaborn-whitegrid')
    plt.scatter(pca[:, 0], pca[:, 1], c=pca[:, 2], cmap='inferno')
    plt.colorbar()
    plt.savefig(fname)
    plt.close()


def main(config):
    traj = {name: read_traj(traj) for name, traj in config.trajectories.items()}

    if hasattr(config, "full_traj"):
        all_traj = read_traj(config.full_traj)
    else:
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
    print(all_soap.shape)
    print(msg)

    tranformer = make_pipeline(PCA(n_components=config.components))

    tranformer = tranformer.fit(all_soap)

    # calculate variance ratios on the merged data
    variance = tranformer.named_steps['pca'].explained_variance_ratio_
    var=np.cumsum(np.round(variance, decimals=3)*100)
    print("PCA (dim={}) variance: {}".format(
        str(tranformer.named_steps['pca'].n_components_),
        var)
    )

    transformed = {name: tranformer.transform(k) for name, k in soap.items()}
    all_pca = tranformer.transform(all_soap)

    np.savetxt("allsoap.pca", all_pca)

    for k, x in transformed.items():
        np.savetxt("{}soap.pca".format(k), x)
        
    if config.plot:
        for k, x in transformed.items():
            np.random.shuffle(x)
            plot(x[:5000, :], "PhaseSpace2D_{}.png".format(k))

    np.random.shuffle(all_pca)
    plot(all_pca[:15000, :], "PhaseSpace2D_all.png")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config", type=str,
                        help="config file")
    args = parser.parse_args()

    main(get_config(args.config, "transform"))