FROM jjanzic/docker-python3-opencv

# RUN pip install torch
# RUN pip install tensorflow==1.15.0

# RUN pip install gym==0.14.0
# RUN pip install numpy==1.18.1
# RUN pip install pyglet==1.3.2
# RUN pip install pygame==1.9.6
# RUN pip install stable-baselines>=2.9.0
# RUN pip install Pillow==6.2.0

# FROM python:3.7.4
MAINTAINER Tae Min Ha

COPY . /marlenv
WORKDIR /marlenv

RUN pip install --trusted-host pypi.python.org -r requirements.txt

# ADD example.py ./

CMD [ "python", "example.py",  "--mode=saveImage" ]

# RUN python example.py --mode=runGUI

