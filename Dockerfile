FROM python:3
MAINTAINER Tae Min Ha

ADD test.py /

RUN pip install stable_baselines
RUN pip install tensorflow==1.15.0 
RUN pip install torch
RUN pip install pygame 
RUN brew install sdl2 sdl2_gfx sdl2_image sdl2_mixer sdl2_net sdl2_ttf
RUN git clone https://github.com/pygame/pygame.git

RUN cd pygame 
RUN python setup.py install

WORKDIR /marlenv

RUN python test.py 
