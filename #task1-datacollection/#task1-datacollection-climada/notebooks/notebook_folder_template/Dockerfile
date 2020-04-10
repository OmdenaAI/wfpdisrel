FROM jupyter/base-notebook:python-3.7.3 as base

USER $NB_UID
WORKDIR $HOME

ENV JUPYTER_ENABLE_LAB=yes

RUN conda update -n base conda

COPY requirements.txt .
RUN conda install --yes --file requirements.txt -c conda-forge \
    && \
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
