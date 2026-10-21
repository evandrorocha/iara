"""
Script para ajustar os caminhos em notebooks/src/*.py e notebooks/*.ipynb
após a movimentação dos códigos geradores para notebooks/src/.
"""
import os
import re
import json

ROOT = r"c:\Users\ev_ro\git\IARA"
SRC_DIR = os.path.join(ROOT, "notebooks", "src")
NB_DIR = os.path.join(ROOT, "notebooks")

# 1. Ajustar generate_notebooks.py
gen_nb_path = os.path.join(SRC_DIR, "generate_notebooks.py")
with open(gen_nb_path, "r", encoding="utf-8") as f:
    c = f.read()

# Substituir o salvamento dos notebooks 1-4
c = c.replace(
    'create_notebook(os.path.join(HERE, "1_comparacao_geral.ipynb"), cells_1)',
    'NOTEBOOKS_DIR = os.path.abspath(os.path.join(HERE, ".."))\ncreate_notebook(os.path.join(NOTEBOOKS_DIR, "1_comparacao_geral.ipynb"), cells_1)'
)
c = c.replace('os.path.join(HERE, "2_opcao_rejeicao.ipynb")', 'os.path.join(NOTEBOOKS_DIR, "2_opcao_rejeicao.ipynb")')
c = c.replace('os.path.join(HERE, "3_cpa_proximidade.ipynb")', 'os.path.join(NOTEBOOKS_DIR, "3_cpa_proximidade.ipynb")')
c = c.replace('os.path.join(HERE, "4_trajetoria_experimentos.ipynb")', 'os.path.join(NOTEBOOKS_DIR, "4_trajetoria_experimentos.ipynb")')

# Substituir bloco sys.path em HELPERS
old_help_path = """if os.path.abspath('.') not in sys.path:
    sys.path.insert(0, os.path.abspath('.'))   # cwd = notebooks/
import iara_results as R"""
new_help_path = """for _p in ('.', 'src', 'notebooks', 'notebooks/src'):
    if os.path.exists(os.path.join(_p, 'iara_results.py')) and os.path.abspath(_p) not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))
import iara_results as R"""
c = c.replace(old_help_path, new_help_path)

with open(gen_nb_path, "w", encoding="utf-8") as f:
    f.write(c)
print("Updated generate_notebooks.py")

# 2. Ajustar generate_demon_notebook.py
gen_demon_path = os.path.join(SRC_DIR, "generate_demon_notebook.py")
with open(gen_demon_path, "r", encoding="utf-8") as f:
    c = f.read()

c = c.replace(
    'def build(filename=os.path.join(HERE, "5_analise_demon.ipynb")):',
    'NOTEBOOKS_DIR = os.path.abspath(os.path.join(HERE, ".."))\n\ndef build(filename=os.path.join(NOTEBOOKS_DIR, "5_analise_demon.ipynb")):'
)
old_demon_setup = "sys.path.insert(0, os.path.abspath('.') if os.path.exists('iara_results.py') else os.path.abspath('notebooks'))\nimport iara_results as R"
new_demon_setup = """for _p in ('.', 'src', 'notebooks', 'notebooks/src'):
    if os.path.exists(os.path.join(_p, 'iara_results.py')) and os.path.abspath(_p) not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))
import iara_results as R"""
c = c.replace(old_demon_setup, new_demon_setup)
with open(gen_demon_path, "w", encoding="utf-8") as f:
    f.write(c)
print("Updated generate_demon_notebook.py")

# 3. Ajustar generate_comparacao_demon_notebook.py
gen_comp_dem_path = os.path.join(SRC_DIR, "generate_comparacao_demon_notebook.py")
with open(gen_comp_dem_path, "r", encoding="utf-8") as f:
    c = f.read()
c = c.replace(
    'def build(filename=os.path.join(HERE, "7_comparacao_modelos_demon.ipynb")):',
    'NOTEBOOKS_DIR = os.path.abspath(os.path.join(HERE, ".."))\n\ndef build(filename=os.path.join(NOTEBOOKS_DIR, "7_comparacao_modelos_demon.ipynb")):'
)
c = c.replace(old_demon_setup, new_demon_setup)
with open(gen_comp_dem_path, "w", encoding="utf-8") as f:
    f.write(c)
print("Updated generate_comparacao_demon_notebook.py")

# 4. Ajustar generate_comparacao_multimodal_notebook.py
gen_comp_multi_path = os.path.join(SRC_DIR, "generate_comparacao_multimodal_notebook.py")
with open(gen_comp_multi_path, "r", encoding="utf-8") as f:
    c = f.read()
c = c.replace(
    'def build(filename=os.path.join(HERE, "8_comparacao_multimodal_mel_demon_lofar.ipynb")):',
    'NOTEBOOKS_DIR = os.path.abspath(os.path.join(HERE, ".."))\n\ndef build(filename=os.path.join(NOTEBOOKS_DIR, "8_comparacao_multimodal_mel_demon_lofar.ipynb")):'
)
c = c.replace(old_demon_setup, new_demon_setup)
with open(gen_comp_multi_path, "w", encoding="utf-8") as f:
    f.write(c)
print("Updated generate_comparacao_multimodal_notebook.py")

# 5. Ajustar os geradores que usam 'notebooks' em loop de caminhos
for fname in sorted(os.listdir(SRC_DIR)):
    if not fname.endswith(".py"):
        continue
    fpath = os.path.join(SRC_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    modified = False
    
    # Substituir for _p in ('.', 'notebooks'):
    if "for _p in ('.', 'notebooks'):" in content:
        content = content.replace(
            "for _p in ('.', 'notebooks'):",
            "for _p in ('.', 'src', 'notebooks', 'notebooks/src'):"
        )
        modified = True
        
    if "for _p in ('.', 'notebooks', 'notebooks/generators'):" in content:
        content = content.replace(
            "for _p in ('.', 'notebooks', 'notebooks/generators'):",
            "for _p in ('.', 'src', 'notebooks', 'notebooks/src'):"
        )
        modified = True
        
    if "for _p in (os.getcwd(), os.path.join(os.getcwd(), 'notebooks')):" in content:
        content = content.replace(
            "for _p in (os.getcwd(), os.path.join(os.getcwd(), 'notebooks')):",
            "for _p in (os.getcwd(), os.path.join(os.getcwd(), 'src'), os.path.join(os.getcwd(), 'notebooks'), os.path.join(os.getcwd(), 'notebooks', 'src')):"
        )
        modified = True

    if modified:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated setup paths in {fname}")

# 6. Atualizar as células dos 25 arquivos .ipynb em notebooks/
SEARCH_PATTERNS = [
    # 10, 12, 13
    (
        "sys.path.insert(0, os.path.abspath('.') if os.path.exists('iara_results.py') else os.path.abspath('notebooks'))",
        "for _p in ('.', 'src', 'notebooks', 'notebooks/src'):\n    if os.path.exists(os.path.join(_p, 'iara_results.py')) and os.path.abspath(_p) not in sys.path:\n        sys.path.insert(0, os.path.abspath(_p))"
    ),
    # 1, 2, 3, 4
    (
        "if os.path.abspath('.') not in sys.path:\n    sys.path.insert(0, os.path.abspath('.'))   # cwd = notebooks/",
        "for _p in ('.', 'src', 'notebooks', 'notebooks/src'):\n    if os.path.exists(os.path.join(_p, 'iara_results.py')) and os.path.abspath(_p) not in sys.path:\n        sys.path.insert(0, os.path.abspath(_p))"
    ),
    # 20, 21, 22, 23
    (
        "sys.path.insert(0, os.path.abspath('.'))\nsys.path.insert(0, os.path.abspath('notebooks'))",
        "for _p in ('.', 'src', 'notebooks', 'notebooks/src'):\n    if os.path.exists(os.path.join(_p, 'iara_results.py')) and os.path.abspath(_p) not in sys.path:\n        sys.path.insert(0, os.path.abspath(_p))"
    ),
    # 24, 25
    (
        "sys.path.insert(0, os.getcwd())",
        "for _p in ('.', 'src', 'notebooks', 'notebooks/src'):\n    if os.path.exists(os.path.join(_p, 'iara_results.py')) and os.path.abspath(_p) not in sys.path:\n        sys.path.insert(0, os.path.abspath(_p))"
    ),
    # 6, 9, 11, 14, 15, 16
    (
        "for _p in ('.', 'notebooks'):",
        "for _p in ('.', 'src', 'notebooks', 'notebooks/src'):"
    ),
    # 17, 18, 19
    (
        "for _p in (os.getcwd(), os.path.join(os.getcwd(), 'notebooks')):",
        "for _p in (os.getcwd(), os.path.join(os.getcwd(), 'src'), os.path.join(os.getcwd(), 'notebooks'), os.path.join(os.getcwd(), 'notebooks', 'src')):"
    )
]

for nb_name in sorted(os.listdir(NB_DIR)):
    if not nb_name.endswith(".ipynb"):
        continue
    nb_path = os.path.join(NB_DIR, nb_name)
    with open(nb_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    nb_modified = False
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            for old_pat, new_pat in SEARCH_PATTERNS:
                if old_pat in src:
                    src = src.replace(old_pat, new_pat)
                    cell["source"] = src.splitlines(keepends=True)
                    nb_modified = True
                    break
    
    if nb_modified:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Updated setup cell in {nb_name}")

print("\nConcluído com sucesso!")
