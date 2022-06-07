#!/bin/bash

cp -r ../src/app app
docker build -t qooba/aimatting:dev -f Dockerfile.dev .
rm -r app
