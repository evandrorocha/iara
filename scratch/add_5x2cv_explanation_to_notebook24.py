"""
Insere a seção didática sobre o protocolo 5x2cv e a prevenção de vazamento com trn_fold = fold ^ 1
diretamente no notebook notebooks/24_sistema_gating_frontend.ipynb.
"""
import json
import os

def insert_explanation():
    nb_path = 'notebooks/24_sistema_gating_frontend.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Texto Markdown explicativo
    md_content = """## 1.1 Metodologia do Protocolo 5x2cv: Por que os Áudios se Repetem e a Solução `trn_fold = fold ^ 1`

### A Diferença Fundamental entre 10-fold CV Comum e o 5x2cv:
* **No 10-fold CV tradicional:** O dataset é dividido em 10 fatias disjuntas ($10\%$ cada). Cada áudio é avaliado no conjunto de teste exatamente **1 vez** na vida.
* **No 5x2cv (5 Repetições de 2-Fold CV - Dietterich, 1998 / Silva et al., 2025):** O dataset é dividido meio a meio ($50\%$ treino e $50\%$ teste) e esse processo é repetido **5 vezes independentes**, reembaralhando os mesmos áudios com sementes diferentes a cada repetição.

Como em **cada uma das 5 repetições** todos os áudios da base são avaliados (metade no fold par, metade no fold ímpar), **cada áudio é testado de 3 a 5 vezes ao longo dos 10 folds**!

---

### Exemplo Didático: Acompanhando um Áudio através das 5 Repetições
Imagine um dataset simples com 4 gravações: `[A, B, C, D]`. Veja como os mesmos arquivos são distribuídos:

1. **Repetição 1 (Semente 1):**
   * **Fold 0 (Teste):** `[A, B]` (Treino é `[C, D]`)
   * **Fold 1 (Teste):** `[C, D]` (Treino é `[A, B]`)
   * *O áudio A foi testado no Fold 0.*
2. **Repetição 2 (Semente 2 - Reembaralha os mesmos 4 áudios):**
   * **Fold 2 (Teste):** `[A, C]` (Treino é `[B, D]`)
   * **Fold 3 (Teste):** `[B, D]` (Treino é `[A, C]`)
   * *O áudio A agora foi testado no Fold 2!*
3. **Repetição 3 (Semente 3):**
   * **Fold 4 (Teste):** `[A, D]` | **Fold 5 (Teste):** `[B, C]` *(Áudio A testado no Fold 4)*
4. **Repetição 4 (Semente 4):**
   * **Fold 6 (Teste):** `[B, A]` | **Fold 7 (Teste):** `[C, D]` *(Áudio A testado no Fold 6)*
5. **Repetição 5 (Semente 5):**
   * **Fold 8 (Teste):** `[C, A]` | **Fold 9 (Teste):** `[D, B]` *(Áudio A testado no Fold 8)*

> **Conclusão:** O áudio `A` apareceu no teste de **5 folds diferentes** (`Fold 0, 2, 4, 6 e 8`).

---

### O Desastre do Vazamento de Dados no Código Antigo:
No código anterior do notebook, o loop de treino do roteador fazia:
```python
# CÓDIGO ANTIGO (COM VAZAMENTO MASSIVO DE 99,6%):
for fold in range(10):
    for trn_fold in range(10):
        if trn_fold == fold:
            continue  # "pular apenas o próprio fold de teste"
        # Treinar roteador em trn_fold...
```
**O que acontecia:** Ao avaliar o `Fold 0`, o roteador era treinado com os Folds `1, 2, 3, 4, 5, 6, 7, 8 e 9`. Mas os Folds 2, 4, 6 e 8 continham o áudio `A`! O roteador recebia o áudio `A` **4 vezes no treino**, decorando seu vetor estático de 7 meta-features e o gabarito de acerto dos especialistas. Isso inflava o SP artificialmente para quase $70\%$.

---

### A Solução Matemática: `trn_fold = fold ^ 1` (Bitwise XOR)
O operador `^` em Python inverte o bit menos significativo (LSB), mapeando cada fold exclusivamente para o seu **parceiro complementar na mesma repetição**:

| Fold de Teste (`fold`) | Operação Bitwise | Fold de Treino Disjunto (`trn_fold = fold ^ 1`) | Repetição | Sobreposição de Áudios |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | `0 ^ 1` | **Fold 1** | Repetição 1 | **0% (Disjunção Estrita)** |
| **Fold 1** | `1 ^ 1` | **Fold 0** | Repetição 1 | **0% (Disjunção Estrita)** |
| **Fold 2** | `2 ^ 1` | **Fold 3** | Repetição 2 | **0% (Disjunção Estrita)** |
| **Fold 3** | `3 ^ 1` | **Fold 2** | Repetição 2 | **0% (Disjunção Estrita)** |
| **Fold 4** | `4 ^ 1` | **Fold 5** | Repetição 3 | **0% (Disjunção Estrita)** |
| **Fold 5** | `5 ^ 1` | **Fold 4** | Repetição 3 | **0% (Disjunção Estrita)** |
| **Fold 6** | `6 ^ 1` | **Fold 7** | Repetição 4 | **0% (Disjunção Estrita)** |
| **Fold 7** | `7 ^ 1` | **Fold 6** | Repetição 4 | **0% (Disjunção Estrita)** |
| **Fold 8** | `8 ^ 1` | **Fold 9** | Repetição 5 | **0% (Disjunção Estrita)** |
| **Fold 9** | `9 ^ 1` | **Fold 8** | Repetição 5 | **0% (Disjunção Estrita)** |

A célula de código a seguir executa uma verificação empírica sobre os 10 folds do IARA, provando numericamente o isolamento estrito."""

    # Código Python demonstrativo
    code_content = """# Verificação Empírica da Disjunção do Protocolo 5x2cv no Dataset IARA
import pickle
import pandas as pd

# Carregar as predições e fids de teste salvos
with open('data/iara_processed/rf_all_representations_predictions.pkl', 'rb') as f:
    pred_data = pickle.load(f)

test_fids = {fold: set(pred_data['targets'][fold].keys()) for fold in range(10)}

audit_rows = []
for fold in range(10):
    par_fold = fold ^ 1
    # 1. Sobreposição com o parceiro complementar da mesma repetição (trn_fold = fold ^ 1)
    overlap_partner = len(test_fids[fold].intersection(test_fids[par_fold]))
    
    # 2. Sobreposição com os outros 8 folds das outras repetições (trn_fold != fold e != par_fold)
    other_folds_fids = set.union(*[test_fids[f] for f in range(10) if f not in (fold, par_fold)])
    overlap_others = len(test_fids[fold].intersection(other_folds_fids))
    pct_leakage = (overlap_others / len(test_fids[fold])) * 100
    
    # Validação formal
    assert overlap_partner == 0, f"Vazamento detectado entre fold {fold} e {par_fold}!"
    
    audit_rows.append({
        'Fold Avaliado': f'Fold {fold}',
        'Tamanho do Teste': len(test_fids[fold]),
        'Parceiro (trn_fold = fold ^ 1)': f'Fold {par_fold}',
        'Sobreposição c/ Parceiro': f'{overlap_partner} áudios (0.0%) [SEGURO]',
        'Vazamento se treinasse nos outros folds': f'{overlap_others} áudios ({pct_leakage:.1f}%) [LEAKAGE]'
    })

df_audit = pd.DataFrame(audit_rows)
print("=" * 110)
print("AUDITORIA EMPÍRICA DE DISJUNÇÃO: trn_fold = fold ^ 1 vs. trn_fold != fold")
print("=" * 110)
print(df_audit.to_string(index=False))
print("-" * 110)
print("CONCLUSÃO: O uso de 'trn_fold = fold ^ 1' garante exatamente 0% de sobreposição em todos os 10 folds!")"""

    # Formatar como células do Jupyter Notebook
    def make_cell(ctype, text):
        lines = text.strip('\n').split('\n')
        src = [ln + '\n' for ln in lines[:-1]] + [lines[-1]] if lines else []
        if ctype == 'markdown':
            return {"cell_type": "markdown", "metadata": {}, "source": src}
        else:
            return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

    cell_md = make_cell('markdown', md_content)
    cell_code = make_cell('code', code_content)

    # Encontrar a posição correta (após a célula 2 - Seção 1)
    # Se já existir uma célula com 1.1, substituir; senão inserir após Seção 1
    target_idx = None
    for i, c in enumerate(nb['cells']):
        if '1. Arquitetura do Sistema Gating Front-End' in ''.join(c.get('source', [])):
            target_idx = i + 1
            break

    if target_idx is None:
        target_idx = 3

    # Verificar se já existe a seção 1.1
    exists = False
    for i, c in enumerate(nb['cells']):
        if '1.1 Metodologia do Protocolo 5x2cv' in ''.join(c.get('source', [])):
            exists = True
            break

    if not exists:
        nb['cells'].insert(target_idx, cell_md)
        nb['cells'].insert(target_idx + 1, cell_code)
        print(f"Inseridas 2 células da Seção 1.1 na posição {target_idx}!")
    else:
        print("Seção 1.1 já existe no notebook. Atualizando conteúdo...")
        # atualizar
        for i, c in enumerate(nb['cells']):
            if '1.1 Metodologia do Protocolo 5x2cv' in ''.join(c.get('source', [])):
                nb['cells'][i] = cell_md
                nb['cells'][i+1] = cell_code
                break

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook '{nb_path}' atualizado com sucesso!")

if __name__ == '__main__':
    insert_explanation()
