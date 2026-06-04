import os
import sys
import collections
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Set standard output encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

def most_common_value(series):
    return collections.Counter(series).most_common(1)[0][0]

def compute_rejection_curve(csv_path):
    if not os.path.exists(csv_path):
        return None
        
    df = pd.read_csv(csv_path)
    
    # Analyze files one by one to get window consensus
    audio_records = []
    for file_name, group in df.groupby('File'):
        targets = group['Target'].values
        predictions = group['Prediction'].values
        
        # True target of this audio file (majority vote of targets)
        true_target = collections.Counter(targets).most_common(1)[0][0]
        
        # Prediction counts
        pred_counts = collections.Counter(predictions)
        most_common_pred, pred_qty = pred_counts.most_common(1)[0]
        
        # Consensus ratio (confidence index)
        consensus_confidence = pred_qty / len(predictions)
        
        audio_records.append({
            'File': file_name,
            'Target': true_target,
            'Prediction': most_common_pred,
            'Confidence': consensus_confidence
        })
        
    df_audio = pd.DataFrame(audio_records)
    total_audios = len(df_audio)
    
    results = {}
    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for t in thresholds:
        # Filter accepted audios
        df_accepted = df_audio[df_audio['Confidence'] >= t]
        accepted_qty = len(df_accepted)
        
        coverage = (accepted_qty / total_audios) * 100
        
        if accepted_qty > 0:
            targets_acc = df_accepted['Target'].values
            preds_acc = df_accepted['Prediction'].values
            
            # Accuracy
            accuracy = (targets_acc == preds_acc).mean() * 100
            
            # SP Index
            cm = confusion_matrix(targets_acc, preds_acc, labels=[0, 1, 2, 3])
            cm_norm = np.zeros((4, 4))
            for r in range(4):
                row_sum = sum(cm[r])
                if row_sum > 0:
                    cm_norm[r] = (cm[r] / row_sum) * 100
            
            recalls = [cm_norm[i][i] for i in range(4)]
            if any(r == 0 for r in recalls):
                sp_index = 0.0
            else:
                sp_index = np.prod(recalls) ** 0.25
        else:
            accuracy = 0.0
            sp_index = 0.0
            
        results[t] = {
            'coverage': coverage,
            'accuracy': accuracy,
            'sp_index': sp_index
        }
        
    return results

def get_model_stats(model_name, is_calibrated=False):
    base_dir = f"results/trainings/tests/{model_name}"
    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Structure to hold metrics for all folds
    # { t: { 'coverage': [], 'accuracy': [], 'sp_index': [] } }
    all_folds_metrics = {t: {'coverage': [], 'accuracy': [], 'sp_index': []} for t in thresholds}
    
    folds_found = 0
    for fold in range(10):
        if is_calibrated:
            csv_path = os.path.join(base_dir, "eval", f"fold_{fold}", f"{model_name}_multiclass_test_b0.4_-0.19_0_0.csv")
        else:
            csv_path = os.path.join(base_dir, "eval", f"fold_{fold}", f"{model_name}_multiclass_test.csv")
            
        curve = compute_rejection_curve(csv_path)
        if curve is not None:
            folds_found += 1
            for t in thresholds:
                all_folds_metrics[t]['coverage'].append(curve[t]['coverage'])
                all_folds_metrics[t]['accuracy'].append(curve[t]['accuracy'])
                all_folds_metrics[t]['sp_index'].append(curve[t]['sp_index'])
                
    if folds_found == 0:
        return None
        
    # Compute mean and standard deviation
    stats = {}
    for t in thresholds:
        stats[t] = {}
        for metric in ['coverage', 'accuracy', 'sp_index']:
            vals = all_folds_metrics[t][metric]
            mean_val = np.mean(vals)
            # Use ddof=1 for sample standard deviation if folds > 1, else 0
            std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            stats[t][metric] = (mean_val, std_val)
            
    stats['folds_found'] = folds_found
    return stats

def format_cell(stats_t, metric):
    if stats_t is None or metric not in stats_t:
        return "      N/D      "
    mean, std = stats_t[metric]
    return f"{mean:5.1f} ± {std:3.1f}%"

def main():
    mel_model = "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0"
    mel_c05_model = "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C0.5"
    lofar_model = "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C0.5"
    hybrid_model = "svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5"
    mlp_model = "mlp"
    mlp_lofar_model = "mlp_lofar"
    mlp_hybrid_model = "mlp_hybrid_pca64"
    
    print("Carregando resultados dos 10 folds...")
    mel_stats = get_model_stats(mel_model, is_calibrated=False)
    mel_c05_stats = get_model_stats(mel_c05_model, is_calibrated=False)
    lofar_stats = get_model_stats(lofar_model, is_calibrated=False)
    mlp_stats = get_model_stats(mlp_model, is_calibrated=False)
    mlp_lofar_stats = get_model_stats(mlp_lofar_model, is_calibrated=False)
    mlp_hybrid_stats = get_model_stats(mlp_hybrid_model, is_calibrated=False)
    hybrid_stats = get_model_stats(hybrid_model, is_calibrated=False)
    calibrated_stats = get_model_stats(hybrid_model, is_calibrated=True)
    
    # Check folds found
    models_to_check = [
        ("SVM MEL (C=2.0)", mel_stats),
        ("SVM MEL (C=0.5)", mel_c05_stats),
        ("SVM LOFAR", lofar_stats),
        ("MLP Baseline (MEL)", mlp_stats)
    ]
    if mlp_lofar_stats is not None:
        models_to_check.append(("MLP Baseline (LOFAR)", mlp_lofar_stats))
    if mlp_hybrid_stats is not None:
        models_to_check.append(("MLP Híbrido", mlp_hybrid_stats))
    models_to_check.extend([
        ("SVM Híbrido", hybrid_stats),
        ("SVM Calibrado", calibrated_stats)
    ])
    
    for name, stats in models_to_check:
        if stats is not None:
            print(f"  {name}: {stats['folds_found']}/10 folds encontrados.")
        else:
            print(f"  {name}: NENHUM fold encontrado!")
            
    print("\n" + "=" * 215)
    print(" ESTUDO DE REJEIÇÃO COMPARATIVO — ANÁLISE COMPLETA DOS 10 FOLDS (MÉDIA ± DESVIO PADRÃO) ")
    print("=" * 215)
    
    # Dynamic header based on mlp_lofar existence
    if mlp_lofar_stats is not None:
        header_model = f"| {'Limiar (t)':<12} | {'1. SVM MEL (Best)':<38} | {'2. SVM LOFAR (Best)':<38} | {'3. MLP MEL':<38} | {'4. MLP LOFAR':<38} | {'5. SVM HÍBRIDO (C=0.5)':<38} | {'6. SVM CALIBRADO (Biased)':<38} |"
        header_metric = f"| {'':<12} | {'Cob. %':<11} {'ACC %':<11} {'SP %':<12} | {'Cob. %':<11} {'ACC %':<11} {'SP %':<12} | {'Cob. %':<11} {'ACC %':<11} {'SP %':<12} | {'Cob. %':<11} {'ACC %':<11} {'SP %':<12} | {'Cob. %':<11} {'ACC %':<11} {'SP %':<12} | {'Cob. %':<11} {'ACC %':<11} {'SP %':<12} |"
        sep_len = 255
    else:
        header_model = f"| {'Limiar (t)':<12} | {'1. SVM MEL (Best)':<38} | {'2. SVM LOFAR (Best)':<38} | {'3. MLP MEL':<38} | {'4. SVM HÍBRIDO (C=0.5)':<38} | {'5. SVM CALIBRADO (Biased)':<38} |"
        header_metric = f"| {'':<12} | {'Cob. %':<11} {'ACC %':<11} {'SP %':<12} | {'Cob. %':<11} {'ACC %':<11} {'SP %':<12} | {'Cob. %':<11} {'ACC %':<11} {'SP %':<12} | {'Cob. %':<11} {'ACC %':<11} {'SP %':<12} | {'Cob. %':<11} {'ACC %':<11} {'SP %':<12} |"
        sep_len = 215
        
    print(header_model)
    print(header_metric)
    print("-" * sep_len)
    
    stats_list = [mel_stats, lofar_stats, mlp_stats]
    if mlp_lofar_stats is not None:
        stats_list.append(mlp_lofar_stats)
    stats_list.extend([hybrid_stats, calibrated_stats])
    
    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    for t in thresholds:
        row_parts = [f"| t >= {t:<6.1f}"]
        
        for stats in stats_list:
            if stats is not None and t in stats:
                st = stats[t]
                cob_str = format_cell(st, 'coverage')
                acc_str = format_cell(st, 'accuracy')
                sp_str = format_cell(st, 'sp_index')
                row_parts.append(f"| {cob_str:<11} {acc_str:<11} {sp_str:<12}")
            else:
                row_parts.append(f"| {'N/D':^11} {'N/D':^11} {'N/D':^12}")
                
        row_parts.append("|")
        print(" ".join(row_parts))
        
    print("=" * sep_len)
    print("Legenda:")
    print("  - Cob. %: Taxa de Cobertura (porcentagem de áudios aceitos, i.e., 100% - Taxa de Descarte)")
    print("  - ACC %:  Acurácia Global sobre as decisões aceitas (sem contar os descartados)")
    print("  - SP %:   Índice SP (Média Geométrica dos recalls) sobre as decisões aceitas")
    print("  - Formato: Média ± Desvio Padrão calculados ao longo dos 10 folds de validação cruzada.")
    print("=" * sep_len)
    
    # Write results as markdown formatted table to the walkthrough / console
    print("\n--- Tabela em Markdown para o Relatório ---")
    md_header = """| Limiar ($t$) | Modelo | Cobertura ($Cob.$) | Acurácia ($ACC$) | Índice SP ($SP$) |
| :--- | :--- | :---: | :---: | :---: |"""
    print(md_header)
    
    model_labels = [
        ("SVM MEL (C=2.0)", mel_stats),
        ("SVM MEL (C=0.5)", mel_c05_stats),
        ("SVM LOFAR (Best)", lofar_stats),
        ("MLP MEL", mlp_stats)
    ]
    if mlp_lofar_stats is not None:
        model_labels.append(("MLP LOFAR", mlp_lofar_stats))
    if mlp_hybrid_stats is not None:
        model_labels.append(("MLP Híbrido", mlp_hybrid_stats))
    model_labels.extend([
        ("SVM Híbrido", hybrid_stats),
        ("SVM Calibrado", calibrated_stats)
    ])
    
    for t in thresholds:
        for label, stats in model_labels:
            if stats is not None and t in stats:
                mean_c, std_c = stats[t]['coverage']
                mean_a, std_a = stats[t]['accuracy']
                mean_s, std_s = stats[t]['sp_index']
                print(f"| $t \\ge {t:.1f}$ | {label} | ${mean_c:.1f} \\pm {std_c:.1f}\\%$ | ${mean_a:.1f} \\pm {std_a:.1f}\\%$ | ${mean_s:.1f} \\pm {std_s:.1f}\\%$ |")
        print("| --- | --- | --- | --- | --- |")

if __name__ == "__main__":
    main()
