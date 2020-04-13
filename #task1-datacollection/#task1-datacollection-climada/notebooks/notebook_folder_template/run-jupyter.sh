#!/bin/bash

NUMBER=$(cat /dev/urandom | tr -dc '0-9' | fold -w 256 | head -n 1 | sed -e 's/^0*//' | head --bytes 4)
if [ "$NUMBER" == "" ]; then
	  NUMBER=0
fi

docker build -t "jupyter-$NUMBER" .
docker run --rm -p 8888:8888 -v "$PWD":/home/jovyan/ "jupyter-$NUMBER"

