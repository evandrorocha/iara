#!/bin/bash
# Regenera e executa os notebooks afetados pela troca da CNN MEL do repositório (cnn) pela do artigo (cnn_mel_artigo)
cd /workspace/IARA
GENS="generate_notebooks generate_comparacao_lofar_mel_notebooks generate_comparacao_demon_notebook generate_comparacao_multimodal_notebook generate_cnn_small_notebook generate_cnn_medium_notebook generate_cnn_large_notebook generate_bigru_notebook generate_erros_comuns_navios_notebook generate_correlacao_entradas_notebook generate_diversidade_multimodelo_notebook generate_estudo_small_notebook generate_ensemble_entradas_notebook generate_cnn_lofar_erros_notebook"
for g in $GENS; do
  echo "$(date -u +%T) gerando $g"
  python notebooks/src/$g.py > /dev/null || echo "FALHA_GERADOR $g"
done
NBS=""
for nb in 1_comparacao_geral 2_opcao_rejeicao 3_cpa_proximidade 4_trajetoria_experimentos 7_comparacao_modelos_mel 6_comparacao_modelos_lofar 9_comparacao_modelos_demon 12_comparacao_multimodal_mel_demon_lofar 22_analise_complementaridade_cnn_small 23_analise_complementaridade_cnn_medium 24_analise_complementaridade_cnn_large 25_avaliacao_multiclasse_bigru 29_concordancia_erros_comuns_navios 30_correlacao_erros_entradas 31_diversidade_multimodelo_entradas 32_estudo_small 33_ensemble_entradas 34_cnn_lofar_artigo_erros; do
  NBS="$NBS notebooks/$nb.ipynb"
done
python scratch/execute_notebooks.py $NBS
echo "$(date -u +%T) fim"
