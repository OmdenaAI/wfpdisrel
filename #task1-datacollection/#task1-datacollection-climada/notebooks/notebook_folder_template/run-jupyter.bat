set /a NUMBER=(%RANDOM%*9999/32768)+1000

set PORT="8888"
for /f %%w in ('git rev-parse --show-prefix') do (
    SET RELATIVE_TO_GIT_ROOT=%%w
)

for /f %%w in ('git rev-parse --show-toplevel') do (
    SET GIT_ROOT=%%w
)

Setlocal EnableDelayedExpansion
call set RELATIVE_TO_GIT_ROOT=!RELATIVE_TO_GIT_ROOT:#=%%%%23!

docker build -t "jupyter-%NUMBER%" --build-arg MAPPED_PORT=%PORT% --build-arg RELATIVE_PATH="%RELATIVE_TO_GIT_ROOT%" .
docker run --rm -p %PORT%:8888 -v "%GIT_ROOT%":/home/jovyan/ "jupyter-%NUMBER%"