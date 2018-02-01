#!/bin/bash
echo "WARNING: This installation method will no longer be supported in the future"
apt-get update
apt-get install -y bzip2 git wget python3-pip python3-yaml
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
export PATH=$PATH:/root/miniconda3/bin/
conda install -y psutil
python3.5 -m pip install pyyaml
