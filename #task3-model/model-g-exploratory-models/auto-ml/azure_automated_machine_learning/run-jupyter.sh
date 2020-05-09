#!/bin/bash

NUMBER=$(cat /dev/urandom | tr -dc '0-9' | fold -w 256 | head -n 1 | sed -e 's/^0*//' | head --bytes 4)
if [ "$NUMBER" == "" ]; then
	  NUMBER=0
fi

PORT="8888"
RELATIVE_TO_GIT_ROOT=$(git rev-parse --show-prefix)
RELATIVE_TO_GIT_ROOT=${RELATIVE_TO_GIT_ROOT//#/%23}

docker build -t "jupyter-$NUMBER" --build-arg MAPPED_PORT=$PORT --build-arg RELATIVE_PATH=$RELATIVE_TO_GIT_ROOT .
docker run --rm --user root -p $PORT:8888 -v "$(git rev-parse --show-toplevel)":/home/jovyan/ "jupyter-$NUMBER"

