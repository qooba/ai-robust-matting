FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
RUN pip install gdown -q
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install git vim python3-opencv -yq
RUN git clone -q https://github.com/PeterL1n/BackgroundMattingV2
RUN cd BackgroundMattingV2 && pip install -r requirements.txt
RUN gdown https://drive.google.com/uc?id=1ErIAsB_miVhYL9GDlYUmfbqlV293mSYf -O model.pth -q
#RUN gdown https://drive.google.com/uc?id=1tCEk8FE3WGrr49cdL8qMCqHptMCAtHRU -O /tmp/src.mp4 -q
#RUN gdown https://drive.google.com/uc?id=1wAR3JjnTO60B_DUr7ruIJj0Z2pcIGkyP -O /tmp/bgr.png -q
RUN pip install qdi fastapi jinja2 aiofiles python-multipart uvicorn[standard] ffmpeg moviepy

WORKDIR /app
COPY ./app /app
EXPOSE 8000
CMD ./start-reload.sh
