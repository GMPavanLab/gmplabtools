import os

import numpy as np

import gmplabtools
from gmplabtools.analysis import DataSampler

ROOT_DIR = "tests/data"


def run_test():

    filename = f"{ROOT_DIR}/dataset.txt"
    ngrid = 50

    dataset = np.loadtxt(filename)
    ds, idx = DataSampler().minmax_sample(dataset, ngrid)
    np.savetxt("sample_grid_indices.txt", idx, fmt="%d")

    pamm_input = dict(
        d=2,
        trajectory=filename,
        ngrid=ngrid,
        readgrid="sample_grid_indices.txt",
        o="results",
        fpoints=0.2,
        merger=0.1,
        bootstrap=64,
        z=0.01,
    )

    pamm = gmplabtools.Pamm(pamm_input)
    pamm.run()

    # check attributes
    attributes = ["predict", "predict_proba", "pk", "mean_k", "cov_k"]
    assert all([hasattr(pamm, attr) for attr in attributes])

    # check files written
    output_files = ["results.pamm", "results.grid", "results.dim", "results.bs"]
    assert all([os.path.isfile(output_file) for output_file in output_files])

    assert pamm.n == 3

    print("Test passed.")


if __name__ == "__main__":

    run_test()
