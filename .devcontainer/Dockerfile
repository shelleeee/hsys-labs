FROM nvidia/cuda:13.3.1-devel-ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    cmake \
    ninja-build \
    pkg-config \
    sudo \
    lsb-release \
    gnupg \
    software-properties-common \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    clang-18 \
    clangd-18 \
    clang-tidy-18 \
    clang-format-18 \
    libc++-18-dev \
    libc++abi-18-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 \
    --slave /usr/bin/clang++ clang++ /usr/bin/clang++-18 \
    --slave /usr/bin/clangd clangd /usr/bin/clangd-18 \
    --slave /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-18 \
    --slave /usr/bin/clang-format clang-format /usr/bin/clang-format-18

RUN curl -L https://github.com/boyter/scc/releases/download/v3.7.0/scc_Linux_x86_64.tar.gz \
    | tar -xzf - -C /usr/local/bin/ scc
RUN chmod +x /usr/local/bin/scc

USER ubuntu
WORKDIR /workspace
