"""
Analisa a distribuição de confiança do especialista binário SMALL vs MEDIUM.
Usa as file IDs do CSV de avaliação para garantir que estamos olhando
exatamente os mesmos arquivos que o framework avaliou no fold_0.
"""
import sys, os, math, csv
sys.path.insert(0, 'src')
sys.path.insert(0, 'test_scripts')

import numpy as np
from collections import defaultdict

import iara.ml.models.base_model as iara_model
import iara.records
import iara.default as iara_default
import iara.ml.dataset as iara_dataset
import iara.ml.experiment as iara_exp
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc
from iara.default import DEFAULT_DIRECTORIES

EXP = 'svm_nystroem_6000_lofar_binary_small_medium_pca64_elasticnet_l1r0.15'
MODEL_PKL = f'results/trainings/tests/{EXP}/model/fold_0/{EXP}_multiclass.pkl'
CSV_TEST  = f'results/trainings/tests/{EXP}/eval/fold_0/{EXP}_multiclass_test.csv'

# ── 1. Carregar modelo ───────────────────────────────────────────────
model = iara_model.BaseModel.load(MODEL_PKL)
print(f'Modelo: n_targets={model.n_targets}')

# ── 2. Ler CSV e aplicar maioria de votos por arquivo ───────────────
file_target = {}
file_preds  = defaultdict(list)

with open(CSV_TEST) as fh:
    for row in csv.DictReader(fh):
        fid  = int(row['File'])
        tgt  = int(row['Target'])
        pred = int(row['Prediction'])
        file_target[fid] = tgt
        file_preds[fid].append(pred)

file_majority = {}
for fid, preds in file_preds.items():
    file_majority[fid] = max(set(preds), key=preds.count)

# Grupos a analisar
tp_med = [fid for fid, t in file_target.items() if t==1 and file_majority[fid]==1]  # MEDIUM correto
fp_med = [fid for fid, t in file_target.items() if t==0 and file_majority[fid]==1]  # SMALL→MEDIUM
tn_sml = [fid for fid, t in file_target.items() if t==0 and file_majority[fid]==0]  # SMALL correto
fn_sml = [fid for fid, t in file_target.items() if t==1 and file_majority[fid]==0]  # MEDIUM→SMALL

print(f'TP_MED={len(tp_med)}  FP_MED={len(fp_med)}  TN_SML={len(tn_sml)}  FN_SML={len(fn_sml)}')
print(f'Total={len(tp_med)+len(fp_med)+len(tn_sml)+len(fn_sml)}')

# ── 3. Construir loader para carregar janelas por file_id ────────────
import pandas as pd

class SmallMediumFilter:
    def apply(self, df):
        return df[pd.to_numeric(df['Length'], errors='coerce') < 100]

binary_collection = iara.records.CustomCollection(
    collection=iara.records.Collection.OS,
    target=iara.records.GenericTarget(
        n_targets=2,
        function=iara_default.Target.classify_row,
        include_others=False
    ),
    filters=SmallMediumFilter(),
    only_sample=False
)
dp = iara_manager.AudioFileProcessor(
    data_base_dir=DEFAULT_DIRECTORIES.data_dir,
    data_processed_base_dir=DEFAULT_DIRECTORIES.process_dir,
    normalization=iara_proc.Normalization.NORM_L2,
    analysis=iara_proc.SpectralAnalysis.LOFAR,
    n_pts=1024, n_overlap=0, decimation_rate=3, n_mels=256, integration_interval=0.512
)

import iara.ml.models.trainer as iara_trn
from svm import SVMNystroemLocalTrainer

config = iara_exp.Config(
    name=EXP,
    dataset=binary_collection,
    dataset_processor=dp,
    output_base_dir=DEFAULT_DIRECTORIES.training_dir + '/tests',
    input_type=iara_dataset.InputType.Window()
)
trainer_dummy = SVMNystroemLocalTrainer(
    training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
    trainer_id=EXP, n_targets=2, n_components=6000,
)
manager = iara_exp.Manager(config, trainer_dummy)
loader  = manager.get_experiment_loader()

# ── 4. Calcular confiança média por arquivo ─────────────────────────
def run_model_on_file(fid):
    """Carrega janelas do arquivo fid, roda o modelo, retorna scores."""
    try:
        loader.pre_load([fid])
        n_win = loader.size_map[fid]
    except Exception:
        return None
    # Carregar janelas via AudioDataset de um único arquivo
    ds = iara_dataset.AudioDataset(loader, config.input_type, [fid])
    samples = ds.get_samples()
    X = samples.cpu().numpy() if hasattr(samples, 'cpu') else np.array(samples)
    if getattr(model, 'normalize', False) and getattr(model, 'scaler', None) is not None:
        X = model.scaler.transform(X)
    if getattr(model, 'pca', False) and getattr(model, 'pca_trans', None) is not None:
        X = model.pca_trans.transform(X)
    X_t = model.nystroem.transform(X)
    scores = model.sgd.decision_function(X_t)
    return scores

def file_stats(fid):
    scores = run_model_on_file(fid)
    if scores is None:
        return None
    probs = 1.0 / (1.0 + np.exp(-scores))
    confs = np.maximum(probs, 1.0 - probs)
    return {
        'mean_score': float(scores.mean()),
        'mean_conf':  float(confs.mean()),
        'pct_medium': float((scores > 0).mean()),  # fração de janelas votando MEDIUM
    }

print('\nCarregando dados e calculando confiança...')
groups_data = {
    'TP_MED (MEDIUM→MEDIUM)': tp_med,
    'FP_MED (SMALL→MEDIUM)':  fp_med,
    'TN_SML (SMALL→SMALL)':   tn_sml,
    'FN_SML (MEDIUM→SMALL)':  fn_sml,
}

group_results = {}
all_file_ids = tp_med + fp_med + tn_sml + fn_sml

# Pre-carrega todos de uma vez para eficiência
print(f'Pre-carregando {len(all_file_ids)} arquivos...')
loader.pre_load(all_file_ids)

for group_label, fids in groups_data.items():
    stats_list = []
    for fid in fids:
        s = file_stats(fid)
        if s:
            stats_list.append(s)
    group_results[group_label] = stats_list

# ── 5. Exibir resultados ────────────────────────────────────────────
def pstats(vals, key):
    v = [s[key] for s in vals]
    if not v: return 'n=0'
    n = len(v)
    mu = sum(v)/n
    sd = math.sqrt(sum((x-mu)**2 for x in v)/n)
    sv = sorted(v)
    return (f'n={n:3d}  μ={mu:.3f}  σ={sd:.3f}'
            f'  p25={sv[n//4]:.3f}  p50={sv[n//2]:.3f}  p75={sv[3*n//4]:.3f}')

print()
print('=== Confiança média por arquivo (voto majoritário) ===')
for label, stats_list in group_results.items():
    print(f'  {label:32s}  {pstats(stats_list, "mean_conf")}')

print()
print('=== Score médio por arquivo (>0 = MEDIUM, <0 = SMALL) ===')
for label, stats_list in group_results.items():
    print(f'  {label:32s}  {pstats(stats_list, "mean_score")}')

print()
print('=== % de janelas votando MEDIUM por arquivo ===')
for label, stats_list in group_results.items():
    print(f'  {label:32s}  {pstats(stats_list, "pct_medium")}')

# ── 6. Sweep de threshold ────────────────────────────────────────────
# Construir lista completa com grupo de cada arquivo
all_items = []
for fid in tp_med: all_items.append(('TP', 1, file_stats(fid)))
for fid in fp_med: all_items.append(('FP', 0, file_stats(fid)))
for fid in tn_sml: all_items.append(('TN', 0, file_stats(fid)))
for fid in fn_sml: all_items.append(('FN', 1, file_stats(fid)))
all_items = [(g, t, s) for g, t, s in all_items if s is not None]

print()
print('=== Sweep de threshold t (conf < t → fallback=MEDIUM no cascade) ===')
print(f'  {"t":>5}  {"TP":>5} {"FP(S→M)":>8} {"FN(M→S)":>8} {"TN":>5}  {"Prec_M":>7} {"Rec_M":>7} {"Rec_S":>7}')
print('  ' + '-'*65)
for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    tp = fp = fn = tn = 0
    for grp, tgt, s in all_items:
        conf = s['mean_conf']
        # Se conf >= t, usa predicao do especialista; se < t, fallback = MEDIUM (cascade)
        if conf >= t:
            pred = 1 if s['mean_score'] > 0 else 0
        else:
            pred = 1  # fallback cascade = MEDIUM
        if tgt==1 and pred==1: tp += 1
        elif tgt==0 and pred==1: fp += 1
        elif tgt==1 and pred==0: fn += 1
        else: tn += 1
    prec  = tp/(tp+fp) if (tp+fp)>0 else 0
    rec_m = tp/(tp+fn) if (tp+fn)>0 else 0
    rec_s = tn/(tn+fp) if (tn+fp)>0 else 0
    print(f'  {t:>5.2f}  {tp:>5} {fp:>8} {fn:>8} {tn:>5}  {prec:>7.3f} {rec_m:>7.3f} {rec_s:>7.3f}')
