set /a NUMBER=(%RANDOM%*max/32768)+min

docker build -t "jupyter-$NUMBER" .
docker run --rm -p 8888:8888 -v "$PWD":/home/jovyan/ "jupyter-$NUMBER"

