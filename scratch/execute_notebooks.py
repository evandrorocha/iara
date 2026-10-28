"""Executa notebooks in-place (cwd = notebooks/). Uso no container:
    docker exec -w /workspace/IARA iara-dev python scratch/execute_notebooks.py notebooks/8_analise_demon.ipynb [...]
"""
import os
import sys

import nbformat
from nbclient import NotebookClient

ok = True
for path in sys.argv[1:]:
    nb = nbformat.read(path, as_version=4)
    try:
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": os.path.dirname(os.path.abspath(path))}}).execute()
        print("OK  ", path)
    except Exception as e:  # noqa: BLE001
        ok = False
        print("FAIL", path, "->", str(e).strip().splitlines()[-1][:300])
    nbformat.write(nb, path)
sys.exit(0 if ok else 1)
