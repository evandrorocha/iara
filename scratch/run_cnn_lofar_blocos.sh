#!/bin/bash
# CNN LOFAR do artigo (test_scripts/cnn_lofar.py) em blocos de 3 folds, um processo por bloco.
# O fold 0 já foi treinado. Rodar no container iara-dev:
#   nohup bash scratch/run_cnn_lofar_blocos.sh > results/trainings/multirep/_logs/cnn_lofar_blocos.log 2>&1 &
cd /workspace/IARA
LOG=results/trainings/multirep/_logs
for B in 1-3 4-6 7-9; do
    echo "$(date -u '+%F %T') início do bloco $B"
    python test_scripts/cnn_lofar.py -F "$B" > "$LOG/cnn_lofar_v2_folds_$B.log" 2>&1
    rc=$?
    echo "$(date -u '+%F %T') fim do bloco $B (código $rc)"
    if [ $rc -ne 0 ]; then
        echo "bloco $B falhou; parando"
        exit $rc
    fi
done
echo "$(date -u '+%F %T') todos os blocos concluídos"
