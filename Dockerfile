FROM continuumio/miniconda3:4.7.12

ENV PATH /opt/conda/envs/gmplabtools/bin:$PATH

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget curl nano git gfortran build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ADD environment.yml /

RUN conda env create --file /environment.yml

RUN echo "source activate gmplabtools" > ~/.bashrc \
    && mkdir /home/gmplabtools

WORKDIR /home/gmplabtools

CMD [ "/bin/bash" ]