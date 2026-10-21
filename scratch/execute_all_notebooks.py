import os
import sys
import glob
import nbformat
from nbclient import NotebookClient

notebooks = [
    f"notebooks/{i}_{name}.ipynb"
    for i, name in [
        (1, "comparacao_geral"),
        (2, "opcao_rejeicao"),
        (3, "cpa_proximidade"),
        (4, "trajetoria_experimentos"),
        (5, "analise_demon"),
        (6, "analise_lofar_mel"),
        (7, "comparacao_modelos_demon"),
        (8, "comparacao_multimodal_mel_demon_lofar"),
        (9, "analise_hht_teste_cego"),
        (10, "analise_complementaridade_small"),
        (11, "comparacao_modelos_hht"),
        (12, "analise_complementaridade_medium"),
        (13, "analise_complementaridade_large"),
        (14, "analise_complementaridade_mlp_small"),
        (15, "analise_complementaridade_mlp_medium"),
        (16, "analise_complementaridade_mlp_large"),
        (17, "analise_complementaridade_rf_small"),
        (18, "analise_complementaridade_rf_medium"),
        (19, "analise_complementaridade_rf_large"),
        (20, "analise_complementaridade_cnn_small"),
        (21, "analise_complementaridade_cnn_medium"),
        (22, "analise_complementaridade_cnn_large"),
        (23, "avaliacao_multiclasse_bigru"),
    ]
]

print(f"Total notebooks to execute: {len(notebooks)}")

for nb_path in notebooks:
    if not os.path.exists(nb_path):
        print(f"SKIP (not found): {nb_path}")
        continue
    
    print(f"Executing {nb_path}...")
    try:
        nb = nbformat.read(nb_path, as_version=4)
        client = NotebookClient(nb, timeout=300, kernel_name='python3')
        client.execute()
        nbformat.write(nb, nb_path)
        print(f"  SUCCESS: {nb_path}")
    except Exception as e:
        print(f"  ERROR executing {nb_path}: {e}")

print("Done all notebook executions!")
