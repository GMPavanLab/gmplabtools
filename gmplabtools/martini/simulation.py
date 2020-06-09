import os
import json
import subprocess
import datetime
from shutil import copyfile

from .parameters import Martini, init_params

THIS_PACKAGE = __file__


class SetupSim:

    FORMAT = '%Y-%m-%d_%H%M%S'

    def __init__(self, config):
        self.config = config
        self.pwd = THIS_PACKAGE
        self.cwd = os.getcwd()
        self.run_time = datetime.datetime.now().strftime(SetupSim.FORMAT)

    @property
    def simulation(self):
        return 'simulation_{run_time}'.format(run_time=self.run_time)

    @property
    def full_path(self):
        return os.path.join(self.cwd, self.simulation)

    def _create_path(self):
        if not os.path.isdir(self.full_path):
            os.mkdir(self.full_path)
        else:
            raise ValueError("Path {} already exists".format(self.full_path))
        return self

    def _create_template(self):
        params = init_params(self.config.params['fields'])
        template = Martini(**params).get_template(self.config.params['template'])
        template_file = os.path.join(self.full_path, 'martini.itp')
        with open(template_file, 'w') as f:
            f.write(template)
        return json.dumps(params)

    def _copy_files(self):
        sim_config = {}
        for k, f in self.config.files.items():
            destination = os.path.join(self.full_path, os.path.basename(f))
            copyfile(f, destination)
            sim_config[k] = destination

        sim_config['sim_params'] = self._create_template()
        sim_config['md_input'] = os.path.join(self.full_path, 'md_input')
        sim_config['md_output'] = os.path.join(self.full_path, 'md_output')
        sim_config.update(self.config.__dict__['simulation'])

        for k, v in self.config.__dict__.items():
            if k not in sim_config:
                sim_config[k] = v

        json.dump(
            sim_config,
            open(os.path.join(self.full_path, 'simulation.json'), 'w'),
            indent=True
        )
        return sim_config

    def setup(self):
        return Simulation(self._create_path()._copy_files())


class Simulation:

    PRAPARE_CMD = "{exec} grompp -c {gro} -f {mdp} -p {top} -n {index} -o {md_input}"
    RUN_CMD = "{exec} mdrun -s {md_input} -nt {n_cpu} -gpu_id {gpu_id} -deffnm {md_output}"

    def __init__(self, param_dict):
        self.param_dict = param_dict

    def _prepare_cmd(self, **kwargs):
        return Simulation.PRAPARE_CMD.format(**{**self.param_dict, **kwargs})

    def _run_cmd(self, **kwargs):
        return Simulation.RUN_CMD.format(**{**self.param_dict, **kwargs})

    def _run(self, command):
        call = subprocess.run(command.split(), stdout=subprocess.PIPE)
        return self

    def prepare(self, **kwargs):
        return self._run(self._prepare_cmd(**kwargs))

    def run(self, **kwargs):
        return self._run(self._run_cmd(**kwargs))
