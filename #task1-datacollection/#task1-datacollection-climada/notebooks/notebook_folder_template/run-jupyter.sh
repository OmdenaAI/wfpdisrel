NUMBER=$(cat /dev/urandom | tr -dc '0-9' | fold -w 256 | head -n 1 | sed -e 's/^0*//' | head --bytes 3)
if [ "$NUMBER" == "" ]; then
	  NUMBER=0
fi

NEW_UUID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)

docker build -t "house-prices-jupyter-$NUMBER" .
docker run --rm -p 8888:8888 -v "$PWD":/home/jovyan/ "house-prices-jupyter-$NUMBER"
