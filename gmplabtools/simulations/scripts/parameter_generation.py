import argparse

import numpy as np
from skopt import Optimizer
from skopt.space import Real

from gmplabtools.shared.config import get_config
from gmplabtools.simulations.parameters import Param


def main(config, n, res):
    dim = len(config.params['fields'])
    parameters = Param(config.params['fields'])

    if res is not None:
        data = np.loadtxt(res)
        opt = Optimizer([Real(0.03, 0.25), Real(0.03, 0.25)],
                        base_estimator="gp",
                        acq_optimizer="auto")

        y = list(data[:, :dim])
        x = [list(row[:dim]) for row in data]
        opt.tell(x, y)
        parameter_list = opt.ask(n, strategy='cl_mean')
    else:
        parameter_list = []
        for i in range(n):
            parameter_list.append([value for field, value in parameters()])

    np.savetxt("sampled_parameters.txt", np.array(parameter_list))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config", type=str,
                        help="config file")
    parser.add_argument("-n", dest="n", type=int,
                        help="sample size")
    parser.add_argument("-r", dest="result", type=str, default=None,
                        help="partial result")
    args = parser.parse_args()
    main(get_config(args.config), args.n, args.result)
