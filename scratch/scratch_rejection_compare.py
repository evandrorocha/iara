import os
import collections
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

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

def main():
    # File paths
    mel_csv = "results/trainings/tests/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0/eval/fold_0/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0_multiclass_test.csv"
    lofar_csv = "results/trainings/tests/svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0/eval/fold_0/svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0_multiclass_test.csv"
    hybrid_csv = "results/trainings/tests/svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5/eval/fold_0/svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5_multiclass_test.csv"
    calibrated_csv = "results/trainings/tests/svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5/eval/fold_0/svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5_multiclass_test_b0.4_-0.19_0_0.csv"
    
    print("=" * 135)
    print(" ESTUDO DE REJEIÇÃO COMPARATIVO — FOLD 0 (TEST SET BY-AUDIO) ")
    print("=" * 135)
    
    mel_curve = compute_rejection_curve(mel_csv)
    lofar_curve = compute_rejection_curve(lofar_csv)
    hybrid_curve = compute_rejection_curve(hybrid_csv)
    calibrated_curve = compute_rejection_curve(calibrated_csv)
    
    # Table header
    print(f"{'Limiar (t)':<12} | {'1. SVM MEL (C=2.0)':<25} | {'2. SVM LOFAR (C=2.0)':<25} | {'3. SVM HÍBRIDO (C=0.5)':<25} | {'4. SVM CALIBRADO (Bias)':<25} |")
    print(f"{'':<12} | {'Cob.%':<5} {'ACC%':<5} {'SP%':<5} | {'Cob.%':<5} {'ACC%':<5} {'SP%':<5} | {'Cob.%':<5} {'ACC%':<5} {'SP%':<5} | {'Cob.%':<5} {'ACC%':<5} {'SP%':<5} |")
    print("-" * 135)
    
    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for t in thresholds:
        row_str = f" t >= {t:<6.1f} | "
        
        # 1. SVM Mel
        if mel_curve is not None and t in mel_curve:
            m = mel_curve[t]
            row_str += f"{m['coverage']:4.1f}% {m['accuracy']:4.1f}% {m['sp_index']:4.1f}% | "
        else:
            row_str += f"{'N/D':^18} | "
            
        # 2. SVM LOFAR
        if lofar_curve is not None and t in lofar_curve:
            lof = lofar_curve[t]
            row_str += f"{lof['coverage']:4.1f}% {lof['accuracy']:4.1f}% {lof['sp_index']:4.1f}% | "
        else:
            row_str += f"{'N/D':^18} | "
            
        # 3. SVM Hybrid
        if hybrid_curve is not None and t in hybrid_curve:
            hy = hybrid_curve[t]
            row_str += f"{hy['coverage']:4.1f}% {hy['accuracy']:4.1f}% {hy['sp_index']:4.1f}% | "
        else:
            row_str += f"{'N/D':^18} | "
            
        # 4. SVM Calibrated
        if calibrated_curve is not None and t in calibrated_curve:
            cal = calibrated_curve[t]
            row_str += f"{cal['coverage']:4.1f}% {cal['accuracy']:4.1f}% {cal['sp_index']:4.1f}% |"
        else:
            row_str += f"{'N/D':^18} |"
            
        print(row_str)
        
    print("=" * 135)
    print("Legenda:")
    print("  - Cob. %: Taxa de Cobertura (porcentagem de áudios classificados, i.e., 100% - Taxa de Descarte)")
    print("  - ACC %:  Acurácia Global sobre as decisões aceitas (sem considerar os rejeitados)")
    print("  - SP %:   Índice SP (Média Geométrica dos recalls) sobre as decisões aceitas")
    print("=" * 135)

if __name__ == "__main__":
    main()
