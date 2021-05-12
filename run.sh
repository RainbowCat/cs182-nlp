#!/usr/bin/env bash

pipenv run python3 main.py --use-bert=False --use-vader=False --use-cnn=False --gpus=0 &
pipenv run python3 main.py --use-bert=False --use-vader=False --use-cnn=True --gpus=1 &
pipenv run python3 main.py --use-bert=False --use-vader=True --use-cnn=False --gpus=2 &
pipenv run python3 main.py --use-bert=False --use-vader=True --use-cnn=True --gpus=3 &
pipenv run python3 main.py --use-bert=True --use-vader=False --use-cnn=False --gpus=4 &
pipenv run python3 main.py --use-bert=True --use-vader=False --use-cnn=True --gpus=5 &
pipenv run python3 main.py --use-bert=True --use-vader=True --use-cnn=False --gpus=6 &
pipenv run python3 main.py --use-bert=True --use-vader=True --use-cnn=True --gpus=7 &
