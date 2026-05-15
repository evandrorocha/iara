@echo off

if "%1" == "up" goto up
if "%1" == "up-gpu" goto up-gpu
if "%1" == "down" goto down
if "%1" == "bash" goto bash
if "%1" == "test-cnn" goto test-cnn
if "%1" == "run-all" goto run-all

echo Use: make [comando]
echo Comandos disponiveis: up, up-gpu, down, bash, test-cnn, run-all
goto end

:up
docker-compose up -d --build
goto end

:up-gpu
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
goto end

:down
docker-compose down
goto end

:bash
docker-compose exec iara bash
goto end

:test-cnn
docker-compose exec iara python test_scripts/cnn.py
goto end

:run-all
docker-compose exec iara bash training_scripts/run_all.sh
goto end

:end
