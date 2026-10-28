"""
Script de geração do notebook 23: notebooks/23_analise_complementaridade_cnn_medium.ipynb
Análise de complementaridade, efeito oráculo e grau de consenso na CNN 2D — classe MEDIUM.
O construtor é comum aos NBs 22-24 (generate_cnn_small_notebook.build_cnn_notebook); todos os números
vêm das predições em disco via notebooks/iara_results.py e notebooks/cnn_complementaridade_comum.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_cnn_small_notebook import build_cnn_notebook  # noqa: E402


def create_notebook(filename="notebooks/23_analise_complementaridade_cnn_medium.ipynb"):
    build_cnn_notebook(1, filename)


if __name__ == "__main__":
    create_notebook()
