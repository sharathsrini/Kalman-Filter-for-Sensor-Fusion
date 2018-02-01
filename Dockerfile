FROM python

MAINTAINER @joshuacook

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh tmp/Miniconda3-latest-Linux-x86_64.sh
RUN bash tmp/Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH $PATH:/root/miniconda3/bin/
COPY environment.yml  .
RUN conda install --yes pyyaml
RUN conda env create -f environment.yml

RUN conda install --name CarND-TensorFlow-Lab -c conda-forge tensorflow

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Copy sample notebooks.
COPY . /notebooks

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /
RUN chmod +x run_jupyter.sh

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

WORKDIR "/notebooks"

CMD ["/run_jupyter.sh"]
