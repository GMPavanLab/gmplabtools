FROM continuumio/miniconda3:4.7.12

ENV CONDA_ENV_PREFIX /opt/conda/envs/gmplabtools
ENV PATH ${CONDA_ENV_PREFIX}/bin:$PATH
ENV LD_LIBRARY_PATH ${CONDA_ENV_PREFIX}/lib:$LD_LIBRARY_PATH

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    wget curl nano git gfortran build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ADD environment.yml /

RUN conda env create --file /environment.yml \
    && conda clean --all -y

RUN echo "source activate gmplabtools" > ~/.bashrc \
    && mkdir /home/gmplabtools \
    && ln -s ${CONDA_ENV_PREFIX}/lib/liblapack.so /usr/lib/liblapack.so \
    && ln -s ${CONDA_ENV_PREFIX}/lib/libblas.so /usr/lib/libblas.so

WORKDIR /home/gmplabtools

CMD [ "/bin/bash" ]