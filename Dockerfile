from nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /root
USER root

ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 서버 관련 유틸
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y ffmpeg wget net-tools build-essential git curl vim nmon tmux lsof && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip

# 파이썬 관련 유틸
RUN pip install -U pip wheel setuptools && \
    pip install k2==1.24.4.dev20240210+cuda12.1.torch2.2.0 -f https://k2-fsa.github.io/k2/cuda.html && \
    pip install transformers==4.42.4 accelerate==0.32.1 datasets==2.20.0 evaluate && \
    pip install scipy sentencepiece deepspeed==0.14.4 wandb && \
    pip install soundfile librosa jiwer torch-audiomentations && \
    pip install setproctitle glances[gpu] && \
    pip install ruff natsort cmake && \
    pip install torch==2.2.0+cu121 torchaudio==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
