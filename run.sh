#!/bin/bash

docker run -it --gpus all -p 8000:8000 --rm --name aimatting -v $(pwd)/src/app:/app qooba/aimatting:dev /bin/bash
