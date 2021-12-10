# GMP Lab Tools

Tools for the analysis of Full Atomistic Simulations of [GMPLab](https://www.gmpavanlab.com/)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Python package dependencies are contained in the environment.yml file from which a conda enviroment can be created.

```
conda env create -f environment.yml
```

### Installing

To install the package the fortran code needs to be compiled.
In order to do that gfortran compiler needs to be installed in the machine.
To compile it is just sufficient to execute.

```
python setup.py compile
```

A pamm executable will be stored in the pamm/bin folder.
The next step after having installed the environment is to install the package into it either via

```
python setup.py install
```

or for development purposes via

```
python setup.py develop
```

#### In case of errors

If `python setup.py compile` fails (for example if blas and lapack are not installed on your system) you can copy the precompiled static executable `pamm_ubuntu` or `pamm_mac` into the pamm bin directory.

```
mkdir -p ./gmplabtools/pamm/precompiledPamm/bin
cp ./gmplabtools/pamm/precompiledPamm/pamm_ubuntu ./gmplabtools/pamm/precompiledPamm/bin/pamm
```
or
```
mkdir -p ./gmplabtools/pamm/precompiledPamm/bin
cp ./gmplabtools/pamm/precompiledPamm/pamm_mac ./gmplabtools/pamm/precompiledPamm/bin/pamm
```


And then install gmplabtools with `python setup.py install`.

### Docker

In order to build docker images following command should be used

```
docker build -t gmplabtools .
```

Running image is pretty straight forward. Following command opens containers terminal.

```
docker run --rm -it -v $PWD:/home/gmplabtools gmplabtools
```

We can also build a pamm by running

```
docker run --rm -v $PWD:/home/gmplabtools gmplabtools python setup.py compile
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

