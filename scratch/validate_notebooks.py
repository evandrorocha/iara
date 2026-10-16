import json
import traceback

notebooks = [
    'notebooks/20_analise_complementaridade_cnn_small.ipynb',
    'notebooks/21_analise_complementaridade_cnn_medium.ipynb',
    'notebooks/22_analise_complementaridade_cnn_large.ipynb'
]

for nb_path in notebooks:
    print(f'=== Validating {nb_path} ===')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    env = {'__name__': '__main__'}
    import matplotlib
    matplotlib.use("Agg")
    
    code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
    for idx, cell in enumerate(code_cells):
        code = ''.join(cell['source'])
        try:
            exec(code, env)
        except Exception as e:
            print(f'Error in cell {idx}: {e}')
            traceback.print_exc()
            break
    else:
        print(f'All {len(code_cells)} code cells in {nb_path} executed cleanly!')
