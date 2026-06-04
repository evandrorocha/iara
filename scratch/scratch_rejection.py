import os
import collections
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

EXP_NAME = "svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5"
BASE_DIR = os.path.abspath(f"results/trainings/tests/{EXP_NAME}")
EVAL_DIR = os.path.join(BASE_DIR, "eval", "fold_0")

class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}

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
    baseline_csv = os.path.join(EVAL_DIR, f"{EXP_NAME}_multiclass_test.csv")
    
    # Look for the optimal bias CSV file in the directory
    # Suffix for bias_small=0.40 and bias_medium=-0.19 is _b0.4_-0.19_0_0
    biased_csv = os.path.join(EVAL_DIR, f"{EXP_NAME}_multiclass_test_b0.4_-0.19_0_0.csv")
    
    print("=" * 115)
    print(" ANÁLISE DE TAXA DE REJEIÇÃO / COBERTURA TEMPORAL — FOLD 0 (TEST SET BY-AUDIO) ")
    print("=" * 115)
    
    baseline_curve = compute_rejection_curve(baseline_csv)
    biased_curve = compute_rejection_curve(biased_csv)
    
    if baseline_curve is None:
        print("Erro: CSV de baseline não encontrado!")
        return
        
    print(f"{'Limiar Confiança (t)':<22} | {'Baseline (Sem Bias)':<28} | {'Calibrado (SMALL=0.40, MEDIUM=-0.19)':<38} | {'Redução Descarte':<12}")
    print(f"{'':<22} | {'Cob. %':<7} {'ACC %':<8} {'SP %':<8} | {'Cob. %':<8} {'ACC %':<8} {'SP %':<8} |")
    print("-" * 115)
    
    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    for t in thresholds:
        base = baseline_curve[t]
        
        # Formatting for display
        base_str = f"{base['coverage']:5.1f}%   {base['accuracy']:5.1f}%   {base['sp_index']:5.1f}%"
        
        if biased_curve is not None and t in biased_curve:
            bi = biased_curve[t]
            bi_str = f"{bi['coverage']:6.1f}%   {bi['accuracy']:5.1f}%   {bi['sp_index']:5.1f}%"
            
            # Discard reduction: coverage_calibrated - coverage_baseline
            cov_diff = bi['coverage'] - base['coverage']
            diff_str = f"{cov_diff:+7.2f}%"
        else:
            bi_str = f"{'N/D':^28}"
            diff_str = f"{'N/D':^12}"
            
        print(f" t >= {t:<16.1f} | {base_str} | {bi_str} | {diff_str}")
        
    print("=" * 115)
    print("Nota: 'Redução Descarte' positiva (+) indica que o modelo calibrado conseguiu classificar MAIS áudios ")
    print("      (rejeitou MENOS) do que o baseline sob o mesmo limiar de confiança, reduzindo a taxa de descarte!")
    print("=" * 115)

if __name__ == "__main__":
    main()
