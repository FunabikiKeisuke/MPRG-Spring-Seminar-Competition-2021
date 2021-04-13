#!/bin/bash

TAG="funabiki/mprg-spring-seminar-competition-2021:1.8.0-cuda11.1-cudnn8-devel"
cd "$(dirname "${0}")/.." || exit

DOCKER_BUILDKIT=1 docker build --progress=plain -t ${TAG} -f docker/Dockerfile .