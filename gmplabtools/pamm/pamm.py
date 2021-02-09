import os

import numpy as np

from gmplabtools.analysis.tools import oracle_shrinkage


class Gauss:
    """
    Class that wraps a normally distributed random variable.
    """
    MIN_VALUE = 1E-20
    LOGP0 = -1E100

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self._cov_det = self.cov_det()

    @property
    def dim(self):
        return len(self.mean)

    @property
    def _valid_covariance(self):
        return np.isnan(self.cov).sum() == 0

    def cov_det(self):
        det = np.linalg.det(self.cov)
        return max(det, Gauss.MIN_VALUE)

    def logpdf(self, x):
        if self._valid_covariance:
            shift = x - self.mean
            a = -np.log(((2 * np.pi) ** self.dim * self._cov_det) ** 0.5)
            b = np.dot(np.dot(shift, np.linalg.inv(self.cov)), shift)
            return a - 0.5 * b
        else:
            return Gauss.LOGP0

    def pdf(self, x):
        return np.exp(self.logpdf(x))


class Pamm:
    """
    Class that allows to interact with pamm and parse the results into python.

    The __init__ methods require you to pass a dictionary with the parameters required by the PAMM
    algorithm. The dictionary must contain the required keys listed in `INPUT_FIELDS`.

    Original implementation: https://github.com/cosmo-epfl/pamm
    Paper: https://pubs.acs.org/doi/10.1021/acs.jctc.7b00993

    Required fields:
        trajectory: filename of the input dataset read by PAMM (txt file format).
        d: n of dimensions of the input dataset.
        ngrid: number of gridpoints for gaussian kde.
        gridfile: filename storing gridpoint files (txt file); if not provide PAMM would select ngrid gridpoints.
        o: basename for output files generation.
        fpoints or fspread: parameters to tune the weighting of the local covariance

    Optional fields:
        bootstrap: number of boostrap iterations; this is required for the calculations of the adjacency matrix.
        merger: merge clusters having occupancy lower than the given probaility threshold.
        qs: Scaling factor for Quick-Shift algorithm.
        nms: number of steps for mean-shift.
    """
    BIN_PATH = os.path.dirname(__file__) + "/bin/pamm"

    INPUT_FIELDS = (
        "d", "bootstrap", "fpoints", "fspread", "qs",
        "o", "gridfile", "trajectory", "nms", "ngrid", "z",
        "merger"
    )

    def __init__(self, input_dict, verbose=True):
        self.input_dict = input_dict
        self.run_status = None
        self.verbose = verbose

        self.p = None
        self.grid = None
        self.cluster = None
        self.weights = None
        self.gmm = None
        self.bs = None

    @property
    def command_parser(self):
        return self.format()

    @property
    def dimension(self):
        return self.input_dict.get("d", None)

    def format(self):
        """
        Checks the format of the input.
        """
        pamm_command = {}
        for k in Pamm.INPUT_FIELDS:
            pamm_command[k] = self.input_dict.get(k, None)

        if not ((pamm_command["fspread"] is None) ^ (pamm_command["fpoints"] is None)):
            raise ValueError("Must provide only one between `fspread` and `fpoints`.")

        if pamm_command["gridfile"] is not None:
            if not os.path.isfile(pamm_command["gridfile"]):
                raise ValueError("Grid file `{}` was not found.".format(pamm_command["gridfile"]))

        if not os.path.isfile(pamm_command["trajectory"]):
            raise ValueError("Trajectory `{}` was not found.".format(pamm_command["trajectory"]))

        fields = [k for k, v in pamm_command.items() if v is not None and k != "trajectory"]
        command = "{exec} {args} {verbose} < {input_traj}".format(
            exec=Pamm.BIN_PATH,
            verbose="" if not self.verbose else "-v",
            args=" ".join(["{}{} {}".format("-", k, pamm_command[k]) for k in fields]),
            input_traj=pamm_command["trajectory"]
        )
        return command

    def run(self):
        """
        Run gmplabtools/pamm/bin/pamm using the paramters in the input_dict and read the results
        """
        command = self.command_parser
        print(f"Executing command: {command}")
        proc = os.system(command)
        self.run_status = proc
        self.read_output()
        return self

    def predict(self, x):
        return self.gmm.predict(x)

    def predict_proba(self, x):
        return self.gmm.predict_proba(x)

    @property
    def _bootstrap_file(self):
        return "{}.{}".format(self.input_dict["o"], "bs")

    @property
    def _pamm_file(self):
        return "{}.{}".format(self.input_dict["o"], "pamm")

    @property
    def _grid_file(self):
        return "{}.{}".format(self.input_dict["o"], "grid")

    @property
    def _dim_file(self):
        return "{}.{}".format(self.input_dict["o"], "dim")

    def read_output(self):
        """
        Methods that reads back the pamm output as numpy array anc create attributes.
        """
        if os.path.isfile(self._bootstrap_file):
            bs = np.loadtxt(self._bootstrap_file).astype(int)
            setattr(self, "bs", bs)
        else:
            msg = "Bootstrap output file {} was not found.".format(self._bootstrap_file)
            FileNotFoundError(msg)

        if os.path.isfile(self._pamm_file):
            gmm = PammGMM.read_clusters(self._pamm_file)
            setattr(self, "gmm", gmm)
        else:
            msg = "Parameter output file {} was not found.".format(self._pamm_file)
            FileNotFoundError(msg)

        if os.path.isfile(self._grid_file):
            grid = np.loadtxt(self._grid_file)
            setattr(self, "grid", grid[:, :self.dimension])
            setattr(self, "cluster", grid[:, self.dimension].astype(int))
            setattr(self, "p", grid[:, self.dimension + 1])
        else:
            msg = "Parameter output file {} was not found.".format(self._grid_file)
            FileNotFoundError(msg)

        if os.path.isfile(self._dim_file):
            dim = np.loadtxt(self._dim_file)
            setattr(self, "dim", dim)
        else:
            msg = "Parameter output file {} was not found.".format(self._dim_file)
            FileNotFoundError(msg)


class PammGMM:
    """
    This class received the GMM parameters from pamm and is used to predict on
    other datesets.
    """
    def __init__(self, pk, mean_k, cov_k):
        self.pk = pk
        self.mean_k = mean_k
        self.cov_k = cov_k
        self.mixtures = [Gauss(m, c) for m, c in zip(mean_k, cov_k)]

    def _predict_proba(self, x):
        log_prob = [mixture.logpdf(x) for mixture in self.mixtures]
        prob = np.array([np.exp(log_p) * self.pk[i] for i, log_p in enumerate(log_prob)])
        return prob

    def predict_proba(self, x):
        """
        Predict cluster assignment probabilities.
        """
        prob = np.apply_along_axis(self._predict_proba, 1, x)
        return prob / np.sum(prob, axis=1).reshape((-1, 1))

    def predict(self, x):
        """
        Predict cluster assignment probabilities.
        """
        return np.argmax(self.predict_proba(x), axis=1)

    @classmethod
    def read_clusters(cls, filename, grid_file=None, bootstrap_file=None):
        """
        Read pamm output parameter and return an instance of GMMPredict class.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError("File {} not found.".format(filename))
        else:
            pk, means, cov, = [], [], []
            zeros = []
            for i, line in enumerate(open(filename, 'r').readlines()):
                if i == 2:
                    D, n_cluster = map(int, line.split())
                if i > 2:
                    values = list(map(float, line.split()))
                    p, m, c = values[0], np.array(values[1:1 + D]), np.array(values[1 + D:]).reshape((D, D))
                    pk.append(p)
                    means.append(m)
                    cov.append(oracle_shrinkage(c, D))

                    if np.isnan(c).sum() != 0:
                        zeros.append(str(i - 2))

            predict = cls(pk, means, cov)

            if grid_file is not None:
                if not os.path.isfile(grid_file):
                    raise FileNotFoundError("File {} not found.".format(grid_file))
                else:
                    grid = np.loadtxt(grid_file)
                    setattr(predict, "grid", grid[:, :D])
                    setattr(predict, "cluster", grid[:, D].astype(int))
                    setattr(predict, "p", grid[:, D + 1])

            if bootstrap_file is not None:
                if not os.path.isfile(bootstrap_file):
                    raise FileNotFoundError("File {} not found.".format(bootstrap_file))
                else:
                    bs = np.loadtxt(bootstrap_file).astype(int)
                    setattr(predict, "bs", bs)

            if zeros:
                msg = ("There are {} clusters with null"
                       " covariance: {}".format(len(zeros), ", ".join(zeros)))
                print(msg)
            return predict