"""
Ensemble de Random Forests por média de probabilidades (soft voting), sobre o 5x2cv oficial.

Não treina nada: lê as probabilidades por janela já gravadas pelos RF de cada representação
(results/trainings/multirep/rf_*_4c/eval/fold_k/*_multiclass_test.csv, colunas Prediction_0..3 = predict_proba)
e combina por gravação:
    p_rep(arquivo) = média das probabilidades das janelas do arquivo no RF da representação
    p(arquivo)     = média de p_rep sobre as representações (pesos iguais)
    predição       = argmax p(arquivo)
A média por arquivo é feita antes da média entre representações porque as janelas não são as mesmas
(MEL/LOFAR em 5 s, HHT/DEMON em 5,12 s). Os folds, e portanto o treino de cada RF, são os mesmos (5x2cv por `ID`).

Variantes:
  * ensemble_rf_prob_mel-lofar-hht-demon_4c -> as 4 representações (proposta original);
  * ensemble_rf_prob_mel-lofar-hht_4c       -> sem DEMON. ATENÇÃO: a exclusão do DEMON foi decidida depois de
    ver os resultados no teste (DEMON SP 44); é uma variante exploratória, não uma escolha pré-registrada.

Só 4 classes: não há RF MEL/LOFAR (mel5s/lofar5s) treinados em 3 classes.
Saída: uma linha por arquivo (n_windows = 1), com Prediction_0..3 = probabilidades do ensemble.

Rodar na raiz do repositório:  python scratch/ensemble_rf_prob_5x2cv.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_writer import ResultsWriter, RESULTS_DIR  # noqa: E402

RF = {'MEL': 'rf_mel5s_n100_balanced_4c', 'LOFAR': 'rf_lofar5s_n100_balanced_4c',
      'HHT': 'rf_hht_n100_balanced_4c', 'DEMON': 'rf_demon_n100_balanced_4c'}
VARIANTES = {
    'ensemble_rf_prob_mel-lofar-hht-demon_4c': ['MEL', 'LOFAR', 'HHT', 'DEMON'],
    'ensemble_rf_prob_mel-lofar-hht_4c': ['MEL', 'LOFAR', 'HHT'],
}
N_CLASSES = 4
PCOLS = [f'Prediction_{c}' for c in range(N_CLASSES)]


def fold_csv(exp, k, kind):
    return os.path.join(RESULTS_DIR, exp, 'eval', f'fold_{k}', f'{exp}_{kind}.csv')


def file_probs(exp, k):
    df = pd.read_csv(fold_csv(exp, k, 'multiclass_test'))
    assert np.allclose(df[PCOLS].sum(axis=1), 1, atol=1e-6), f'{exp} fold {k}: Prediction_k não são probabilidades'
    return df.groupby('File').agg(Target=('Target', 'first'), **{c: (c, 'mean') for c in PCOLS})


def train_files(exp, k):
    s = pd.read_csv(fold_csv(exp, k, 'split'))
    s = s[s.Subset == 'trn']
    return dict(zip(s.File.astype(int), s.Target.astype(int)))


def main():
    for name, reps in VARIANTES.items():
        exps = [RF[r] for r in reps]
        writer = ResultsWriter(name, script=__file__, params={
            'model': 'média das probabilidades por arquivo dos RF (soft voting, pesos iguais)',
            'members': {r: RF[r] for r in reps},
            'combination': 'p_rep = média das janelas do arquivo; p = média de p_rep; predição = argmax',
            'rows': 'uma linha por arquivo (n_windows = 1)',
            'num_classes': N_CLASSES,
            'post_hoc_choice': 'sem DEMON escolhido após ver o teste' if 'DEMON' not in reps else None,
        })
        for k in range(10):
            P = [file_probs(e, k) for e in exps]
            tr = [train_files(e, k) for e in exps]
            assert all(t == tr[0] for t in tr[1:]), f'fold {k}: treino diferente entre os RF'
            idx = P[0].index
            for p in P[1:]:
                assert set(p.index) == set(idx), f'fold {k}: arquivos de teste diferentes entre os RF'
                assert (p.loc[idx, 'Target'] == P[0].Target).all()
            prob = sum(p.loc[idx, PCOLS].values for p in P) / len(P)
            writer.save_fold(k, idx.values, P[0].Target.values, prob.argmax(1), train_files=tr[0], scores=prob)
        print(f'{name}: 10 folds gravados ({len(idx)} arquivos de teste no fold 9)')


if __name__ == '__main__':
    main()
