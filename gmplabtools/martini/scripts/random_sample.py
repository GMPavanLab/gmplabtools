import argparse

import numpy as np

from gmplabtools.shared.config import get_config
from gmplabtools.martini.parameters import Param


def main(config, n):
    parameters = Param(config.params['fields'])
    parameter_list = []
    for i in range(n):
        parameter_list.append([value for field, value in parameters()])

    np.savetxt('filename.txt', np.array(parameter_list))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config", type=str,
                        help="config file")
    parser.add_argument("-n", dest="n", type=int,
                        help="sample size")
    args = parser.parse_args()
    main(get_config(args.config), args.n)
