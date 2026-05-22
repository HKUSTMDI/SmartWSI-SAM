FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
       iputils-ping net-tools curl vim wget gcc g++ make \
       openssh-client libssl-dev git \
       libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV TIME_ZONE=Asia/Shanghai
RUN echo "${TIME_ZONE}" > /etc/timezone && ln -sf /usr/share/zoneinfo/${TIME_ZONE} /etc/localtime

ENV CONDA_AUTO_UPDATE_CONDA=false
ENV CONDA_DIR=/opt/miniconda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p $CONDA_DIR \
    && rm ~/miniconda.sh

# Python 3.10 env -> onnx 1.14.1 / opencv have prebuilt wheels, no source build
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda create -y -n app python=3.10 \
    && conda clean -afy
ENV PATH=$CONDA_DIR/envs/app/bin:$PATH

WORKDIR /app
COPY . .

RUN python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && python -m pip --no-cache-dir install --upgrade pip \
    && pip install -e .

VOLUME ["/app/models"]
ENV SERVER_PORT=9091
EXPOSE ${SERVER_PORT}
CMD ["/bin/bash","./bin/start.sh"]