#FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ARG PIP_EXTRA_INDEX_URL

RUN apt-get -y update && apt-get -y install git

# Install Apex library.
RUN git clone https://github.com/NVIDIA/apex
WORKDIR apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install main stuff and copy code over
RUN pip install --ignore-installed certifi sklearn scikit-image pyyaml h5py pytorch-metric-learning colorlog
RUN pip install --extra-index-url $PIP_EXTRA_INDEX_URL eai-shuriken

WORKDIR /src
RUN mkdir /src/results && chmod 777 /src/results
COPY . .

RUN mkdir /home/beckhamc && chmod 777 /home/beckhamc


