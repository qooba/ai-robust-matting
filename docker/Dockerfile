FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install git curl python3-opencv -yq
RUN git clone -q 
RUN cd RobustVideoMatting && pip install -r requirements_inference.txt && cd .. && rm -rf RobustVideoMatting
RUN pip install qdi fastapi jinja2 aiofiles python-multipart uvicorn[standard] ffmpeg moviepy
WORKDIR /app/models
RUN curl -LO https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth
RUN curl -LO https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth
WORKDIR /app
COPY ./app /app
EXPOSE 8000
CMD ./start-reload.sh
