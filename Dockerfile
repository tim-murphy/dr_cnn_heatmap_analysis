# FROM tensorflow/tensorflow:latest-gpu
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip python-is-python3\
	libgl1-mesa-glx vim git

RUN groupadd --gid 1000 cnn
RUN useradd --uid 1000 --gid 1000 --home-dir /home/cnn cnn
RUN usermod --append --groups sudo cnn
RUN mkdir /home/cnn && chown cnn:cnn /home/cnn

USER cnn
ENV PATH="$PATH:/home/cnn/.local/bin"
ENV TF_CPP_MIN_LOG_LEVEL=1
RUN pip install matplotlib tensorflow[and-cuda]==2.19.0 opencv-python-headless\
 	tqdm
RUN git config --global --add safe.directory /cnn

WORKDIR /cnn/src
# ENTRYPOINT ["python", "train_thyself.py", "refer"]
