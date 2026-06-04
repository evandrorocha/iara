import os
import sys
import collections
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

def compute_rejection_curve(csv_path):
    if not os.path.exists(csv_path):
        return None
        
    df = pd.read_csv(csv_path)
    
    audio_records = []
    for file_name, group in df.groupby('File'):
        targets = group['Target'].values
        predictions = group['Prediction'].values
        
        true_target = collections.Counter(targets).most_common(1)[0][0]
        pred_counts = collections.Counter(predictions)
        most_common_pred, pred_qty = pred_counts.most_common(1)[0]
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
        df_accepted = df_audio[df_audio['Confidence'] >= t]
        accepted_qty = len(df_accepted)
        coverage = (accepted_qty / total_audios) * 100
        
        if accepted_qty > 0:
            targets_acc = df_accepted['Target'].values
            preds_acc = df_accepted['Prediction'].values
            accuracy = (targets_acc == preds_acc).mean() * 100
            
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

def main():
    model_name = "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C0.5"
    base_dir = f"results/trainings/tests/{model_name}"
    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    all_folds_metrics = {t: {'coverage': [], 'accuracy': [], 'sp_index': []} for t in thresholds}
    
    folds_found = 0
    for fold in range(10):
        # Calibrated filename pattern
        csv_path = os.path.join(base_dir, "eval", f"fold_{fold}", f"{model_name}_multiclass_test_b0.32_0.27_0_0.csv")
        curve = compute_rejection_curve(csv_path)
        if curve is not None:
            folds_found += 1
            for t in thresholds:
                all_folds_metrics[t]['coverage'].append(curve[t]['coverage'])
                all_folds_metrics[t]['accuracy'].append(curve[t]['accuracy'])
                all_folds_metrics[t]['sp_index'].append(curve[t]['sp_index'])
                
    if folds_found == 0:
        print("Erro: Nenhum arquivo CSV de teste calibrado encontrado!")
        return
        
    print("=" * 100)
    print(f" RESULTADOS MULTIFOLD DO SVM LOFAR C=0.5 CALIBRADO (biases: SMALL=0.32, MEDIUM=0.27) ")
    print("=" * 100)
    print(f"{'Limiar (t)':<12} | {'Cobertura (%)':<18} | {'Acurácia (%)':<18} | {'Índice SP (%)':<18}")
    print("-" * 100)
    for t in thresholds:
        cov_mean = np.mean(all_folds_metrics[t]['coverage'])
        cov_std = np.std(all_folds_metrics[t]['coverage'], ddof=1) if folds_found > 1 else 0.0
        
        acc_mean = np.mean(all_folds_metrics[t]['accuracy'])
        acc_std = np.std(all_folds_metrics[t]['accuracy'], ddof=1) if folds_found > 1 else 0.0
        
        sp_mean = np.mean(all_folds_metrics[t]['sp_index'])
        sp_std = np.std(all_folds_metrics[t]['sp_index'], ddof=1) if folds_found > 1 else 0.0
        
        print(f"t >= {t:.1f}      | {cov_mean:5.1f} ± {cov_std:3.1f}%     | {acc_mean:5.1f} ± {acc_std:3.1f}%     | {sp_mean:5.1f} ± {sp_std:3.1f}%")
    print("=" * 100)

if __name__ == "__main__":
    main()
