#!/bin/bash

docker run -it --gpus all --rm --name aimatting -v $(pwd)/src/app:/app qooba/aimatting:dev /bin/bash
