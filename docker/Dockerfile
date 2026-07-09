FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    # MX250 (sm_61) + A100 (sm_80) + H100 (sm_90)
    TORCH_CUDA_ARCH_LIST="6.1;8.0;9.0" \
    MAX_JOBS=4

# Locale
RUN apt-get update && apt-get install -y --no-install-recommends \
        locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 + all system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && add-apt-repository -y universe \
    && apt-get update && apt-get install -y --no-install-recommends \
        # Python 3.12
        python3.12 \
        python3.12-dev \
        # Build tools
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        gfortran \
        # Version control
        git \
        git-lfs \
        # Networking + multi-node SSH
        curl \
        wget \
        rsync \
        net-tools \
        openssh-client \
        openssh-server \
        pdsh \
        # InfiniBand / RDMA (multi-node)
        ibverbs-utils \
        libibverbs-dev \
        librdmacm-dev \
        rdma-core \
        # Editors + shell utils
        vim \
        tmux \
        screen \
        tree \
        jq \
        lsof \
        unzip \
        ca-certificates \
        # System monitoring
        htop \
        iotop \
        ncdu \
        psmisc \
        procps \
        strace \
        # GPU monitoring
        nvtop \
        # Android Debug Bridge
        android-tools-adb \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default python/python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip for python3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && python3.12 -m pip install --upgrade pip setuptools wheel

# Install PyTorch before anything else — flash-attn compiles against it
RUN pip install --no-cache-dir \
    torch==2.4.1+cu124 \
    torchvision==0.19.1+cu124 \
    torchaudio==2.4.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# flash-attn requires torch at compile time, so --no-build-isolation
RUN pip install --no-cache-dir flash-attn==2.6.3 --no-build-isolation

# All remaining requirements (torch/flash-attn already satisfied, pip will skip)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# SSH config for multi-node training (passwordless root login within cluster)
RUN mkdir -p /var/run/sshd /root/.ssh \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo "StrictHostKeyChecking no" >> /root/.ssh/config \
    && chmod 600 /root/.ssh/config

WORKDIR /workspace

EXPOSE 22

CMD ["/bin/bash"]
