#!/bin/bash
# CNN MEL do artigo (test_scripts/cnn_mel_artigo.py) em blocos de folds, um processo por bloco.
# Rodar no container iara-dev:
#   nohup bash scratch/run_cnn_mel_artigo_blocos.sh > results/trainings/multirep/_logs/cnn_mel_artigo_blocos.log 2>&1 &
cd /workspace/IARA
LOG=results/trainings/multirep/_logs
for B in 0-3 4-6 7-9; do
    echo "$(date -u '+%F %T') início do bloco $B"
    python test_scripts/cnn_mel_artigo.py -F "$B" > "$LOG/cnn_mel_artigo_folds_$B.log" 2>&1
    rc=$?
    echo "$(date -u '+%F %T') fim do bloco $B (código $rc)"
    if [ $rc -ne 0 ]; then
        echo "bloco $B falhou; parando"
        exit $rc
    fi
done
echo "$(date -u '+%F %T') todos os blocos concluídos"
