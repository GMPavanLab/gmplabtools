import os
import json
import subprocess
import datetime
from shutil import copyfile

from .parameters import Martini, Param

THIS_PACKAGE = __file__


class SetupSim:

    FORMAT = '%Y-%m-%d_%H%M%S'

    def __init__(self, config):
        self.config = config
        self.pwd = THIS_PACKAGE
        self.cwd = os.getcwd()
        self.run_time = ''

    @property
    def simulation(self):
        if not self.run_time:
            msg = 'The `run_time` attribute was not set and simulation setup can not be generated.'
            raise ValueError(msg)
        return 'simulation_{run_time}'.format(run_time=self.run_time)

    @property
    def full_path(self):
        return os.path.join(self.cwd, self.simulation)

    def _init_path(self):
        self.run_time = datetime.datetime.now().strftime(SetupSim.FORMAT)
        if not os.path.isdir(self.full_path):
            os.mkdir(self.full_path)
        else:
            raise ValueError("Path {} already exists".format(self.full_path))
        return self

    def _init_template(self, config):
        template = Martini(**config).get_template(self.config.params['template'])
        template_file = os.path.join(self.full_path, 'simulations.itp')
        return template_file, template

    def _generate_simulation(self, params):
        template_file, template = self._init_template(params)
        with open(template_file, 'w') as f:
            f.write(template)

        sim_config = {}
        for k, f in self.config.files.items():
            destination = os.path.join(self.full_path, os.path.basename(f))
            copyfile(f, destination)
            sim_config[k] = destination

        sim_config['sim_params'] = params
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

    def __iter__(self):
        for params in Param.set_config(self.config.params['fields'],
                                       self.config.params['parameter_file']):
            self._init_path()
            sim_config = self._generate_simulation(params)
            yield Simulation(sim_config)


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
        _ = subprocess.run(command.split(), stdout=subprocess.PIPE)
        return self

    def prepare(self, **kwargs):
        return self._run(self._prepare_cmd(**kwargs))

    def run(self, **kwargs):
        return self._run(self._run_cmd(**kwargs))
