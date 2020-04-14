FROM jupyter/base-notebook:python-3.7.3 as base

USER $NB_UID
WORKDIR $HOME

ENV JUPYTER_ENABLE_LAB=yes

RUN conda update -n base conda

COPY requirements.txt .
RUN conda install --yes --file requirements.txt -c conda-forge -c anaconda -c pypi\
    && \
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

ARG RELATIVE_PATH=""
ENV RELATIVE_PATH=${RELATIVE_PATH}
ARG MAPPED_PORT="8888"
ENV MAPPED_PORT=${MAPPED_PORT}
ENV JUPYTER_TOKEN="1"

CMD echo "\n==> use url found below:\nhttp://127.0.0.1:$MAPPED_PORT/lab/tree/README.md?file-browser-path=$RELATIVE_PATH&token=1 \n" && start-notebook.sh 
