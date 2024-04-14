FROM python:3.10.7
RUN apt update
RUN apt-get update \
  && apt-get -y install tesseract-ocr 
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN mkdir /fastapi_app
WORKDIR /fastapi_app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r /fastapi_app/requirements.txt

COPY . .

RUN chmod a+x docker/*.sh
