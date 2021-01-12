import os
import sys
import subprocess
from setuptools import setup, Command, find_packages

import gmplabtools


class CleanCommand(Command):
    """Custom clean command to clean up the project root."""
    description = 'clean package root folder'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system("rm -vrf ./cache./cache ./**/.cache "
                  "build "
                  "dist "
                  "./.eggs "
                  "./**/.egg "
                  "./*.egg-info "
                  "./**/*.egg-info "
                  "./gmplabtools/pamm/src/*.o "
                  "./gmplabtools/pamm/src/*.mod "
                  "**/*pyc "
                  "**/__pycache__ "
                  "**.coverage "
                  "**/cover* "
                  ".pytest_cache "
                  "./mypy_cache "
                  )


class TypecheckCommand(Command):
    """Run mypy type checker."""
    description = 'run Mypy on Python source files'
    user_options = [
        ("packages=", None, "Additional packages to combine with options from setup.cfg")
    ]

    def initialize_options(self):
        self.packages = ""

    def finalize_options(self):
        if len(self.packages) > 0:
            package_list = ["--package=" + package for package in self.packages]
            package_list = " ".join(package_list)

    def run(self):
        subprocess.run("python setup.py clean")
        cmd_to_run = "python setup.py -a mypy"
        print("Running command {cmd_to_run}".format(cmd_to_run=cmd_to_run))

        completed_process = subprocess.run(cmd_to_run, universal_newlines=True, shell=True)
        return_code = completed_process.returncode
        if return_code != 0:
            print("Error: Mypy setup.py command has failed", file=sys.stderr)
            sys.exit(0)
        else:
            print("Success: Mypy hasn't discovered any problem.")


class CompileCommand(Command):
    """Compile Pamm fortran code."""
    description = 'Compile pamm fortan code.'
    user_options = []

    def initialize_options(self):
        self.packages = ""

    def finalize_options(self):
        pass

    def run(self):
        # clean folder
        _ = subprocess.run("cd ./gmplabtools/pamm/src && make clean", shell=True)

        cmd_to_run = "cd ./gmplabtools/pamm/src && make"
        completed_process = subprocess.run(cmd_to_run, universal_newlines=True, shell=True)
        return_code = completed_process.returncode

        if return_code != 0:
            print("Failed: Failed to compile Pamm code", file=sys.stderr)
            sys.exit(0)
        else:
            print("Success: Pamm code was compiled.")


setup(
    name="gmplabtools",
    version=gmplabtools.__version__,
    packages=find_packages(),
    package_data={'': ['pamm/bin/*']},
    include_package_data=True,
    # add dependent module if needed
    install_requires=[],
    # add/change tests requirements if needed
    tests_require=[
        "pytest"
    ],
    cmdclass={
        'typecheck': TypecheckCommand,
        'cleanall': CleanCommand,
        'compile': CompileCommand
    }
)
