import json
import matplotlib
matplotlib.use("Agg")

notebooks = [
    "notebooks/14_analise_complementaridade_mlp_small.ipynb",
    "notebooks/15_analise_complementaridade_mlp_medium.ipynb",
    "notebooks/16_analise_complementaridade_mlp_large.ipynb"
]

for nb_name in notebooks:
    print(f"=== Testing {nb_name} ===")
    with open(nb_name, "r", encoding="utf-8") as f:
        nb = json.load(f)
    globs = {"__name__": "__main__"}
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            try:
                exec(code, globs)
            except Exception as e:
                print(f"FAILED on cell {idx}: {e}")
                raise
    print(f"PASS: {nb_name} executed without error!\n")

print("All 3 notebooks passed cell validation with 100% success!")
