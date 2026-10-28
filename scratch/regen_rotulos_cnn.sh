#!/bin/bash
# Regenera e executa os notebooks afetados pela correção dos rótulos da CNN MEL e dos MLPs do repositório
cd /workspace/IARA
GENS="generate_notebooks generate_lofar_mel_notebook generate_comparacao_lofar_mel_notebooks generate_comparacao_demon_notebook generate_comparacao_hht_notebook generate_comparacao_multimodal_notebook generate_mlp_small_notebook generate_mlp_medium_notebook generate_mlp_large_notebook generate_cnn_small_notebook generate_cnn_medium_notebook generate_cnn_large_notebook generate_bigru_notebook generate_gating_notebook generate_super_oraculo_notebook generate_erros_comuns_navios_notebook generate_correlacao_entradas_notebook generate_diversidade_multimodelo_notebook"
for g in $GENS; do
  echo "$(date -u +%T) gerando $g"
  python notebooks/src/$g.py > /dev/null || echo "FALHA_GERADOR $g"
done
for nb in 1_comparacao_geral 2_opcao_rejeicao 3_cpa_proximidade 4_trajetoria_experimentos 5_analise_lofar_mel 6_comparacao_modelos_lofar 7_comparacao_modelos_mel 9_comparacao_modelos_demon 11_comparacao_modelos_hht 12_comparacao_multimodal_mel_demon_lofar 16_analise_complementaridade_mlp_small 17_analise_complementaridade_mlp_medium 18_analise_complementaridade_mlp_large 22_analise_complementaridade_cnn_small 23_analise_complementaridade_cnn_medium 24_analise_complementaridade_cnn_large 25_avaliacao_multiclasse_bigru 26_sistema_gating_frontend 27_analise_falhas_super_oraculo 29_concordancia_erros_comuns_navios 30_correlacao_erros_entradas 31_diversidade_multimodelo_entradas; do
  echo "$(date -u +%T) executando $nb"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/$nb.ipynb > /dev/null 2>notebooks/.err_$nb.txt && { echo "OK $nb"; rm -f notebooks/.err_$nb.txt; } || echo "FALHA $nb (ver notebooks/.err_$nb.txt)"
done
echo "$(date -u +%T) fim"
