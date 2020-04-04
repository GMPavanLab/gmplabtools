FROM continuumio/miniconda3:4.7.12

USER root

ADD environment.yml /

RUN conda env create --file /environment.yml

RUN echo "source activate gmplabtools" > ~/.bashrc

ENV PATH /opt/conda/envs/gmplabtools/bin:$PATH

WORKDIR /

CMD [ "/bin/bash" ]