set /a NUMBER=(%RANDOM%*9999/32768)+1000

set PORT="8888"
for /f w in ('git rev-parse --show-toplevel') do set RELATIVE_TO_GIT_ROOT=%cd:w=""%}

set RELATIVE_TO_GIT_ROOT=${RELATIVE_TO_GIT_ROOT//#/%23}

docker build -t "jupyter-%NUMBER%" --build-arg MAPPED_PORT=%PORT% --build-arg RELATIVE_PATH=%RELATIVE_TO_GIT_ROOT% .
docker run --rm -p %PORT%:8888 -v "$(git rev-parse --show-toplevel)":/home/jovyan/ "jupyter-%NUMBER%"


