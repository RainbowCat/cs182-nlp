#!/usr/bin/env bash

# b c v
# -fff
# -fft
# tff
# tft
# ttf
# ttt

pipenv run python3 main.py --use-bert=True --use-cnn=False --use-vader=False --gpu-id=0
pipenv run python3 main.py --use-bert=True --use-cnn=False --use-vader=True --gpu-id=0
pipenv run python3 main.py --use-bert=True --use-cnn=False --use-vader=False --gpu-id=0
pipenv run python3 main.py --use-bert=True --use-cnn=True --use-vader=True --gpu-id=0
