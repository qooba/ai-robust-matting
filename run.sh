#!/bin/bash

docker run -it --gpus all -p 8000:8000 --rm --name aimatting qooba/aimatting:robust
