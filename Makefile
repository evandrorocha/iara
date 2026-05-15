.PHONY: up up-gpu down bash run-all test-cnn

# Inicia o container em modo CPU / Mac M1
up:
	docker-compose up -d --build

# Inicia o container em modo GPU (NVIDIA)
up-gpu:
	docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# Desliga e remove o container
down:
	docker-compose down

# Entra no terminal interativo do container
bash:
	docker-compose exec iara bash

# Roda o script de testes da CNN
test-cnn:
	docker-compose exec iara python test_scripts/cnn.py

# Roda todos os treinamentos do script run_all.sh
run-all:
	docker-compose exec iara bash training_scripts/run_all.sh
