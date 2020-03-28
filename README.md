# GMP Lab Tools

Tools for the analysis of Full Atomistic Simulations of [GMPLab](https://www.gmpavanlab.com/)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Python package dependencies are contained in the requirement.yaml files from which a conda enviroment can be created.
```
conda env create -f environment.yml
```

### Installing

To install the packace the fotran code needs to be compiled. In order to do that gfortran compiler needs to
be installed in the machine. To compile it is just sufficient to execute

```
python setup.py build
```

a pamm executable will be stored in the pamm/bin folder. The next step after having installed the enviromnet
is to install the package into it either via


```
python setup.py install
```

oor for developmental purposed via

```
python setup.py develop
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


