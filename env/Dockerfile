FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
# requirements
ADD source.list /etc/apt/sources.list
RUN apt-get update && apt-get install -y python3.10 python3-pip
# torch
COPY torch-2.2.0+cu121-cp310-cp310-linux_x86_64.whl torch-2.2.0+cu121-cp310-cp310-linux_x86_64.whl
RUN pip3 install torch-2.2.0+cu121-cp310-cp310-linux_x86_64.whl
# llama factory requirements
RUN pip3 install transformers==4.38.2 datasets==2.16.1 accelerate==0.27.2 peft==0.9.0 trl==0.7.11 gradio==3.50.2 \
    deepspeed==0.13.1 modelscope ipython scipy einops sentencepiece protobuf jieba rouge-chinese nltk sse-starlette matplotlib \
    --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
# FlashAttention
RUN apt install -y git
RUN pip install ninja -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install packaging -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install flash-attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
# add sklearn
RUN pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple