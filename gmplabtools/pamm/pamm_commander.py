import os
import logging

import numpy as np

from gmplabtools.pamm.lib.tools import GMMPredict


logging.getLogger(__name__).setLevel(level=logging.INFO)


class PammCommander:

    BIN_PATH = os.path.dirname(__file__) + "/bin/pamm"

    INPUT_FIELDS = (
        "d", "bootstrap", "fpoints", "fspread", "qs",
        "o", "readgrid", "trajectory", "nms", "ngrid", "z",
        "merger"
    )

    def __init__(self, input_dict):
        self.input_dict = input_dict
        self.run_status = None

    @property
    def command_parser(self):
        return PammCommander.format(self.input_dict)

    @property
    def dimension(self):
        return self.input_dict.get("d", None)

    @staticmethod
    def format(input_dict):
        """
        Checks the format of the input.
        """
        pamm_command = {}
        for k in PammCommander.INPUT_FIELDS:
            pamm_command[k] = input_dict.get(k, None)

        if not ((pamm_command["fspread"] is None) ^ (pamm_command["fpoints"] is None)):
            raise ValueError("Cannot provide both `fspread` and `fpoints` values.")

        if not os.path.isfile(pamm_command["readgrid"]):
            raise ValueError("Grid file {} was not found.".format(pamm_command["readgrid"]))

        if not os.path.isfile(pamm_command["trajectory"]):
            raise ValueError("Trajectory {} was not found.".format(pamm_command["trajectory"]))

        fields = [k for k, v in pamm_command.items() if v is not None and k != "trajectory"]
        command = "{exec} {args} -v < {input_traj}".format(
            exec=PammCommander.BIN_PATH,
            args=" ".join(["{}{} {}".format("-", k, pamm_command[k]) for k in fields]),
            input_traj=pamm_command["trajectory"]
        )
        return command

    def run(self):
        command = self.command_parser
        logging.info("Executing command: {}".format(" ".join(command)))
        proc = os.system(command)
        self.run_status = proc
        self.read_output()
        return self

    @property
    def bootstrap_file(self):
        return "{}.{}".format(self.input_dict["o"], "bs")

    @property
    def pamm_file(self):
        return "{}.{}".format(self.input_dict["o"], "pamm")

    @property
    def grid_file(self):
        return "{}.{}".format(self.input_dict["o"], "grid")

    @property
    def weights_file(self):
        return "{}.{}".format(self.input_dict["o"], "weights")

    def read_output(self):
        if os.path.isfile(self.bootstrap_file):
            bs =  np.loadtxt(self.bootstrap_file).astype(int)
            setattr(self, "bs", bs)
        else:
            msg = "Bootstrap output file {} was not found.".format(self.bootstrap_file)
            FileNotFoundError(msg)

        if os.path.isfile(self.pamm_file):
            gmm =  GMMPredict.read_clusters(self.pamm_file)
            setattr(self, "gmm", gmm)
        else:
            msg = "Parameter output file {} was not found.".format(self.pamm_file)
            FileNotFoundError(msg)

        if os.path.isfile(self.grid_file):
            grid =  np.loadtxt(self.grid_file)
            setattr(self, "grid", grid[:, :self.dimension])
            setattr(self, "cluster", grid[:, self.dimension ].astype(int))
            setattr(self, "p", grid[:, self.dimension + 1])
        else:
            msg = "Parameter output file {} was not found.".format(self.grid_file)
            FileNotFoundError(msg)

        if os.path.isfile(self.weights_file):
            weights =  np.loadtxt(self.weights_file)
            setattr(self, "weights", weights)
        else:
            msg = "Parameter output file {} was not found.".format(self.weights_file)
            FileNotFoundError(msg)