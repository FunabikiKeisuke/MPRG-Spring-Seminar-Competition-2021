#!/bin/bash

TAG="funabiki/mprg-spring-seminar-competition-2021:1.8.0-cuda11.1-cudnn8-devel"
PROJECT_DIR="$(cd "$(dirname "${0}")/.." || exit; pwd)"

docker run -it --rm \
  --gpus all \
  -v "${PROJECT_DIR}:/work" \
  -w "/work" \
  "${TAG}"
