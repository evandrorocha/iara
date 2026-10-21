"""
Gera e analisa as predições do modelo Random Forest (RF - 100 árvores, balanced) nas 4 representações:
- MEL (256-D)
- LOFAR (1024-D)
- HHT (256-D)
- DEMON (253-D)

Sob o protocolo oficial 5x2cv (10 folds com isolamento rígido por Ship ID).
Calcula a análise de consenso, teto do oráculo, resgates exclusivos e padrões de erro
para as 3 classes de embarcações:
- SMALL (Classe 0)
- MEDIUM (Classe 1)
- LARGE (Classe 2)

Gera os 3 gráficos oficiais de 4 quadrantes para reports/ e artefatos.
"""

import os
import shutil
import time
import pickle
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sk_metrics
import scipy.stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results_writer import ResultsWriter

def mode(x):
    return Counter(x).most_common(1)[0][0]

def compute_sp_index(targets, preds, num_classes=4):
    cm = sk_metrics.confusion_matrix(targets, preds, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    acc = float((targets == preds).mean() * 100)
    return sp, acc, recalls * 100, cm

def get_official_folds(eval_base_dir="results/trainings/tests/svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0/eval"):
    folds = []
    for fold_idx in range(10):
        fold_dir = os.path.join(eval_base_dir, f"fold_{fold_idx}")
        trn_path = os.path.join(fold_dir, "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0_multiclass_trn.csv")
        tst_path = os.path.join(fold_dir, "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0_multiclass_test.csv")
        df_trn = pd.read_csv(trn_path)
        df_tst = pd.read_csv(tst_path)
        folds.append({
            "fold_idx": fold_idx,
            "train_files": df_trn.groupby("File")["Target"].first().to_dict(),
            "test_files": df_tst.groupby("File")["Target"].first().to_dict(),
        })
    return folds

def build_fold_matrices(feature_cache, file_dict):
    X_list, y_list, file_ids = [], [], []
    for fid, target in file_dict.items():
        if fid not in feature_cache:
            continue
        windows = feature_cache[fid]
        X_list.append(windows)
        y_list.extend([target] * len(windows))
        file_ids.extend([fid] * len(windows))
    return np.vstack(X_list), np.array(y_list, dtype=np.int64), np.array(file_ids, dtype=np.int64)

def run_random_forest(feature_cache, folds, num_classes=4, n_estimators=100, random_state=42, writer=None):
    preds_by_fold = {}
    for fold in folds:
        fold_idx = fold["fold_idx"]
        t0 = time.time()
        X_train, y_train, _ = build_fold_matrices(feature_cache, fold["train_files"])
        X_test, y_test, test_fids = build_fold_matrices(feature_cache, fold["test_files"])

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        win_preds = rf.predict(X_test)
        if writer is not None:
            writer.save_fold(fold_idx, test_fids, y_test, win_preds,
                             train_files=fold["train_files"], scores=rf.predict_proba(X_test))

        file_preds = defaultdict(list)
        for fid, pred in zip(test_fids, win_preds):
            file_preds[fid].append(pred)

        preds_by_fold[fold_idx] = {
            fid: int(np.argmax(np.bincount(file_preds[fid], minlength=num_classes)))
            for fid in file_preds
        }
        print(f"  Fold {fold_idx} concluído em {time.time()-t0:.1f}s.")
    return preds_by_fold

def analyze_class(df_class, class_name, target_id):
    total = len(df_class)
    total_correct = (df_class["num_hits"] > 0).sum()

    print("\n" + "=" * 80)
    print(f"ANÁLISE RANDOM FOREST — CLASSE {class_name} ({total} ÁUDIOS DE TESTE NOS 10 FOLDS)")
    print("=" * 80)

    # 1. Desempenho Individual
    print("\n--- 1. DESEMPENHO INDIVIDUAL NO RANDOM FOREST ---")
    ind_recalls = {}
    for name, col in [("MEL", "hit_mel"), ("LOFAR", "hit_lofar"), ("HHT", "hit_hht"), ("DEMON", "hit_demon")]:
        c = df_class[col].sum()
        pct = c / total * 100
        ind_recalls[name] = pct
        print(f"  Acertos RF {name:<6}: {c:4d} / {total} ({pct:5.2f}%)")

    print(f"\n  >>> TETO DO ORÁCULO RF : {total_correct:4d} / {total} ({total_correct/total*100:5.2f}%)")
    print(f"  >>> ERRO COMUM RF      : {total - total_correct:4d} / {total} ({(total - total_correct)/total*100:5.2f}%)")

    # 2. Grau de Consenso
    print("\n--- 2. GRAU DE CONSENSO SOBRE O TOTAL ---")
    c4 = (df_class["num_hits"] == 4).sum()
    c3 = (df_class["num_hits"] == 3).sum()
    c2 = (df_class["num_hits"] == 2).sum()
    c1 = (df_class["num_hits"] == 1).sum()
    c0 = (df_class["num_hits"] == 0).sum()

    print(f"  * 4 Entradas (Unanimidade): {c4:4d} ({c4/total*100:5.2f}%)")
    print(f"  * 3 Entradas (Alto)       : {c3:4d} ({c3/total*100:5.2f}%)")
    print(f"  * 2 Entradas (Dividido)   : {c2:4d} ({c2/total*100:5.2f}%)")
    print(f"  * 1 Entrada  (Exclusivo)  : {c1:4d} ({c1/total*100:5.2f}%)")
    print(f"  * 0 Entradas (Erro Comum) : {c0:4d} ({c0/total*100:5.2f}%)")

    # 3. Resgates Exclusivos
    print("\n--- 3. RESGATES EXCLUSIVOS (Apenas 1 entrada salva o áudio) ---")
    only_mel = ((df_class['hit_mel']) & (~df_class['hit_lofar']) & (~df_class['hit_hht']) & (~df_class['hit_demon'])).sum()
    only_lofar = ((~df_class['hit_mel']) & (df_class['hit_lofar']) & (~df_class['hit_hht']) & (~df_class['hit_demon'])).sum()
    only_hht = ((~df_class['hit_mel']) & (~df_class['hit_lofar']) & (df_class['hit_hht']) & (~df_class['hit_demon'])).sum()
    only_demon = ((~df_class['hit_mel']) & (~df_class['hit_lofar']) & (~df_class['hit_hht']) & (df_class['hit_demon'])).sum()

    print(f"  Salvos EXCLUSIVAMENTE pelo MEL  : {only_mel:4d} ({only_mel/total*100:5.2f}%)")
    print(f"  Salvos EXCLUSIVAMENTE pelo LOFAR: {only_lofar:4d} ({only_lofar/total*100:5.2f}%)")
    print(f"  Salvos EXCLUSIVAMENTE pelo HHT  : {only_hht:4d} ({only_hht/total*100:5.2f}%)")
    print(f"  Salvos EXCLUSIVAMENTE pelo DEMON: {only_demon:4d} ({only_demon/total*100:5.2f}%)")

    # 4. Padrão de Erro
    print(f"\n--- 4. PADRÃO DE CONFUSÃO EM {class_name} ---")
    class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "AMBIENTAL"}
    err_patterns = {}
    for mod_name, col in [("MEL", "pred_mel"), ("LOFAR", "pred_lofar"), ("HHT", "pred_hht"), ("DEMON", "pred_demon")]:
        errs = df_class[df_class[col] != target_id][col].map(class_map).value_counts()
        total_errs = (df_class[col] != target_id).sum()
        print(f"\n  Padrão de Erro no {mod_name} (Total Erros: {total_errs}):")
        err_dist = {}
        for cls, cnt in errs.items():
            pct = cnt / total_errs * 100 if total_errs > 0 else 0
            err_dist[cls] = pct
            print(f"    -> Confundido com {cls:<10}: {cnt:4d} ({pct:5.2f}%)")
        err_patterns[mod_name] = err_dist

    # 5. Votação por Consenso
    print(f"\n--- 5. VOTAÇÃO POR MAIORIA / CONSENSO EM {class_name} ---")
    print(f"  * Unanimidade (4 de 4)    : {c4:4d} ({c4/total*100:5.2f}% de recall)")
    print(f"  * Maioria Estrita (>= 3/4): {c4+c3:4d} ({(c4+c3)/total*100:5.2f}% de recall)")
    print(f"  * Maioria Simples (>= 2/4): {c4+c3+c2:4d} ({(c4+c3+c2)/total*100:5.2f}% de recall)")
    print(f"  * Oráculo (>= 1/4)        : {total_correct:4d} ({total_correct/total*100:5.2f}% de recall)")

    return {
        "total": total,
        "total_correct": total_correct,
        "c4": c4, "c3": c3, "c2": c2, "c1": c1, "c0": c0,
        "only_mel": only_mel, "only_lofar": only_lofar, "only_hht": only_hht, "only_demon": only_demon,
        "ind_recalls": ind_recalls,
        "err_patterns": err_patterns,
    }

def plot_4panel_figure(res, class_name, out_filename, artifact_dir):
    total = res["total"]
    total_corr = res["total_correct"]
    oracle_pct = total_corr / total * 100

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 15,
        'figure.dpi': 150
    })

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))

    # Quadrante 1: Desempenho Isolado vs Oráculo
    ax1 = axes[0, 0]
    reps = ['DEMON\n(253-D)', 'HHT Puro\n(256-D)', 'LOFAR\n(1024-D)', 'MEL\n(256-D)', 'ORÁCULO RF\n(União 4 Modos)']
    rec_vals = [
        res["ind_recalls"]["DEMON"],
        res["ind_recalls"]["HHT"],
        res["ind_recalls"]["LOFAR"],
        res["ind_recalls"]["MEL"],
        oracle_pct
    ]
    colors1 = ['#95a5a6', '#9b59b6', '#e67e22', '#2980b9', '#27ae60']
    b1 = ax1.bar(reps, rec_vals, color=colors1, width=0.55, edgecolor='black', linewidth=0.8)
    b1[-1].set_hatch('//')
    best_ind = max(rec_vals[:-1])
    best_rep = ["DEMON", "HHT", "LOFAR", "MEL"][np.argmax(rec_vals[:-1])]

    ax1.axhline(oracle_pct, color='#27ae60', linestyle='--', lw=1.5, alpha=0.7, label=f'Teto do Oráculo ({oracle_pct:.1f}%)')
    ax1.axhline(best_ind, color='#e67e22', linestyle=':', lw=1.5, alpha=0.7, label=f'Melhor Isolado {best_rep} ({best_ind:.1f}%)')

    ax1.set_title(f'A) Recall RF em {class_name}: Modelos Isolados vs. Teto do Oráculo', fontweight='bold', pad=12)
    ax1.set_ylabel(f'Recall na Classe {class_name} (%)')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper left', frameon=True, fontsize=9.5)

    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                     fontsize=10.5, fontweight='bold', color='#111')

    # Quadrante 2: Pirâmide de Consenso
    ax2 = axes[0, 1]
    labels_cons = ['4 Entradas\n(Unânime)', '3 Entradas\n(Alto Consenso)', '2 Entradas\n(Dividido)', '1 Entrada\n(Exclusivo/Frágil)', '0 Entradas\n(Erro Comum)']
    counts_cons = [res["c4"], res["c3"], res["c2"], res["c1"], res["c0"]]
    pcts_cons = [c / total * 100 for c in counts_cons]
    colors_cons = ['#27ae60', '#2980b9', '#f39c12', '#e74c3c', '#7f8c8d']

    b2 = ax2.bar(labels_cons, counts_cons, color=colors_cons, edgecolor='black', linewidth=0.8, width=0.55)
    ax2.set_title(f'B) Grau de Consenso RF em {class_name} (Total: {total} Áudios)', fontweight='bold', pad=12)
    ax2.set_ylabel(f'Quantidade de Áudios de {class_name}')
    ax2.set_ylim(0, max(counts_cons) * 1.18)

    for bar, pct in zip(b2, pcts_cons):
        h = bar.get_height()
        ax2.annotate(f'{h} ({pct:.1f}%)', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                     fontsize=9.5, fontweight='bold')

    # Quadrante 3: Resgates Exclusivos
    ax3 = axes[1, 0]
    excl_reps = ['HHT Exclusivo', 'MEL Exclusivo', 'LOFAR Exclusivo', 'DEMON Exclusivo']
    excl_counts = [res["only_hht"], res["only_mel"], res["only_lofar"], res["only_demon"]]
    excl_pcts = [c / total * 100 for c in excl_counts]
    colors_excl = ['#9b59b6', '#2980b9', '#e67e22', '#16a085']

    b3 = ax3.barh(excl_reps, excl_counts, color=colors_excl, height=0.5, edgecolor='black', linewidth=0.8)
    ax3.set_title(f'C) Resgates Exclusivos RF em {class_name} (Total: {sum(excl_counts)} Áudios Únicos)', fontweight='bold', pad=12)
    ax3.set_xlabel('Áudios Salvos Exclusivamente por Apenas UMA Entrada')
    ax3.set_xlim(0, max(excl_counts) * 1.35 if max(excl_counts) > 0 else 50)

    champ_idx = np.argmax(excl_counts)
    for idx, (bar, pct) in enumerate(zip(b3, excl_pcts)):
        w = bar.get_width()
        tag = f"  <-- CAMPEÃO DE RESGATES EM {class_name}" if idx == champ_idx else ""
        ax3.annotate(f'{w} áudios ({pct:.1f}%){tag}', xy=(w, bar.get_y() + bar.get_height()/2),
                     xytext=(5, 0), textcoords="offset points", ha='left', va='center',
                     fontsize=9.5, fontweight='bold', color='#111' if idx != champ_idx else '#117864')

    # Quadrante 4: Padrão de Erro
    ax4 = axes[1, 1]
    reps_err = ['LOFAR', 'MEL', 'HHT', 'DEMON']
    other_classes = [c for c in ["SMALL", "MEDIUM", "LARGE", "AMBIENTAL"] if c != class_name]

    c_colors = {'SMALL': '#3498db', 'MEDIUM': '#f39c12', 'LARGE': '#2ecc71', 'AMBIENTAL': '#95a5a6'}
    bottom = np.zeros(len(reps_err))
    w = 0.52

    for other_c in other_classes:
        vals = [res["err_patterns"][r].get(other_c, 0.0) for r in reps_err]
        ax4.bar(reps_err, vals, w, bottom=bottom, label=f'Confundido com {other_c}', color=c_colors.get(other_c, '#888'), alpha=0.9, edgecolor='black', linewidth=0.6)
        for i, val in enumerate(vals):
            if val > 6:
                ax4.text(i, bottom[i] + val/2, f'{val:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        bottom += np.array(vals)

    ax4.set_title(f'D) Padrão de Erro RF em {class_name}: Para onde vai a confusão? (%)', fontweight='bold', pad=12)
    ax4.set_ylabel('Distribuição Percentual dos Erros (%)')
    ax4.set_ylim(0, 118)
    ax4.legend(loc='upper right', frameon=True, fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    plt.savefig(out_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo com sucesso em '{out_filename}'.")

    # Copiar para diretório de artefatos
    artifact_copy = os.path.join(artifact_dir, os.path.basename(out_filename))
    shutil.copy2(out_filename, artifact_copy)
    print(f"Copiado para os artefatos: '{artifact_copy}'.")

def main():
    cache_pred_file = "data/iara_processed/rf_all_representations_predictions.pkl"
    folds = get_official_folds()

    if os.path.exists(cache_pred_file):
        print(f"Carregando predições Random Forest do cache '{cache_pred_file}'...")
        with open(cache_pred_file, "rb") as f:
            data = pickle.load(f)
        mel_preds = data["mel"]
        lofar_preds = data["lofar"]
        hht_preds = data["hht"]
        demon_preds = data["demon"]
        targets_by_fold = data["targets"]
    else:
        # Carregar caches
        print("Carregando cache MEL e DEMON de 'data/iara_processed/cache_fusion_mel_demon.pkl'...")
        with open("data/iara_processed/cache_fusion_mel_demon.pkl", "rb") as f:
            fused_data = pickle.load(f)
        cache_mel = fused_data["mel"]
        cache_demon = fused_data["demon"]

        print("Carregando cache HHT de 'data/iara_processed/cache_hht.pkl'...")
        with open("data/iara_processed/cache_hht.pkl", "rb") as f:
            cache_hht = pickle.load(f)

        print("Carregando cache LOFAR de 'data/iara_processed/cache_lofar.pkl'...")
        with open("data/iara_processed/cache_lofar.pkl", "rb") as f:
            cache_lofar = pickle.load(f)

        targets_by_fold = {f["fold_idx"]: f["test_files"] for f in folds}

        # 1. Random Forest no MEL
        print("\n--- Treinando Random Forest no MEL (256-D, 10 Folds 5x2cv) ---")
        mel_preds = run_random_forest(cache_mel, folds, writer=ResultsWriter(
            "rf_mel5s_n100_balanced_4c", script=__file__,
            params={"model": "RandomForestClassifier", "representation": "MEL", "window": "5.12 s (média de 10 frames de 0.512 s)", "dim": 256,
                    "n_estimators": 100, "class_weight": "balanced", "random_state": 42, "num_classes": 4}))

        # 2. Random Forest no LOFAR
        print("\n--- Treinando Random Forest no LOFAR (1024-D, 10 Folds 5x2cv) ---")
        lofar_preds = run_random_forest(cache_lofar, folds, writer=ResultsWriter(
            "rf_lofar5s_n100_balanced_4c", script=__file__,
            params={"model": "RandomForestClassifier", "representation": "LOFAR", "window": "5.12 s (média de 10 frames de 0.512 s)", "dim": 1024,
                    "n_estimators": 100, "class_weight": "balanced", "random_state": 42, "num_classes": 4}))

        # 3. Random Forest no HHT
        print("\n--- Treinando Random Forest no HHT (256-D, 10 Folds 5x2cv) ---")
        hht_preds = run_random_forest(cache_hht, folds, writer=ResultsWriter(
            "rf_hht_n100_balanced_4c", script=__file__,
            params={"model": "RandomForestClassifier", "representation": "HHT", "dim": 256,
                    "n_estimators": 100, "class_weight": "balanced", "random_state": 42, "num_classes": 4}))

        # 4. Random Forest no DEMON
        print("\n--- Treinando Random Forest no DEMON (253-D, 10 Folds 5x2cv) ---")
        demon_preds = run_random_forest(cache_demon, folds, writer=ResultsWriter(
            "rf_demon_n100_balanced_4c", script=__file__,
            params={"model": "RandomForestClassifier", "representation": "DEMON", "dim": 253,
                    "n_estimators": 100, "class_weight": "balanced", "random_state": 42, "num_classes": 4}))

        # Salvar predições
        with open(cache_pred_file, "wb") as f:
            pickle.dump({
                "mel": mel_preds,
                "lofar": lofar_preds,
                "hht": hht_preds,
                "demon": demon_preds,
                "targets": targets_by_fold
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Predições Random Forest salvas com sucesso em '{cache_pred_file}'.")

    # Montar DataFrame consolidado
    records = []
    for fold_idx in range(10):
        t_dict = targets_by_fold[fold_idx]
        for fid, tgt in t_dict.items():
            pm = mel_preds[fold_idx].get(fid, -1)
            pl = lofar_preds[fold_idx].get(fid, -1)
            ph = hht_preds[fold_idx].get(fid, -1)
            pd_ = demon_preds[fold_idx].get(fid, -1)

            hm = (pm == tgt)
            hl = (pl == tgt)
            hh = (ph == tgt)
            hd = (pd_ == tgt)

            records.append({
                "fold": fold_idx,
                "file_id": fid,
                "target": tgt,
                "pred_mel": pm,
                "pred_lofar": pl,
                "pred_hht": ph,
                "pred_demon": pd_,
                "hit_mel": hm,
                "hit_lofar": hl,
                "hit_hht": hh,
                "hit_demon": hd,
                "num_hits": hm + hl + hh + hd
            })

    df_all = pd.DataFrame(records)

    # 1. Analisar SMALL (Target 0)
    df_sml = df_all[df_all["target"] == 0].copy()
    res_sml = analyze_class(df_sml, "SMALL", 0)

    # 2. Analisar MEDIUM (Target 1)
    df_med = df_all[df_all["target"] == 1].copy()
    res_med = analyze_class(df_med, "MEDIUM", 1)

    # 3. Analisar LARGE (Target 2)
    df_lrg = df_all[df_all["target"] == 2].copy()
    res_lrg = analyze_class(df_lrg, "LARGE", 2)

    # Salvar resultados compilados
    out_results = "data/iara_processed/rf_complementarity_results.pkl"
    with open(out_results, "wb") as f:
        pickle.dump({
            "df_all": df_all,
            "res_sml": res_sml,
            "res_med": res_med,
            "res_lrg": res_lrg,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nResultados consolidados salvos com sucesso em '{out_results}'.")

    # Gerar os 3 gráficos oficiais de 4 quadrantes
    artifact_dir = r"C:\Users\ev_ro\.gemini\antigravity-ide\brain\ffb9500e-86ad-47f3-9cf6-d5e7cd24de4f"
    plot_4panel_figure(res_sml, "SMALL", "reports/analise_complementaridade_rf_small_oraculo.png", artifact_dir)
    plot_4panel_figure(res_med, "MEDIUM", "reports/analise_complementaridade_rf_medium_oraculo.png", artifact_dir)
    plot_4panel_figure(res_lrg, "LARGE", "reports/analise_complementaridade_rf_large_oraculo.png", artifact_dir)

if __name__ == "__main__":
    main()
