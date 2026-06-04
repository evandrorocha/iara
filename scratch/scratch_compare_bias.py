import os
import collections
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

EXP_NAME = "svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5"
BASE_DIR = os.path.abspath(f"results/trainings/tests/{EXP_NAME}")
EVAL_DIR = os.path.join(BASE_DIR, "eval", "fold_0")

class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}

def most_common_value(series):
    return collections.Counter(series).most_common(1)[0][0]

def compute_metrics(csv_path):
    if not os.path.exists(csv_path):
        return None
        
    df = pd.read_csv(csv_path)
    
    # Majority vote by audio
    df_audio = df.groupby('File').agg({
        'Target': most_common_value,
        'Prediction': most_common_value
    }).reset_index()
    
    targets = df_audio['Target'].values
    predictions = df_audio['Prediction'].values
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions, labels=[0, 1, 2, 3])
    
    # Normalized confusion matrix
    cm_norm = np.zeros((4, 4))
    for r in range(4):
        row_sum = sum(cm[r])
        if row_sum > 0:
            cm_norm[r] = (cm[r] / row_sum) * 100
            
    # Class Recalls (diagonal of normalized matrix)
    recalls = [cm_norm[i][i] for i in range(4)]
    
    # SP Index (Geometric Mean of Recalls)
    sp_index = np.prod(recalls) ** 0.25
    
    # Overall Accuracy
    accuracy = (targets == predictions).mean() * 100
    
    return {
        'accuracy': accuracy,
        'sp_index': sp_index,
        'recalls': recalls,
        'cm_norm': cm_norm
    }

def main():
    baseline_csv = os.path.join(EVAL_DIR, f"{EXP_NAME}_multiclass_test.csv")
    biased_csv = os.path.join(EVAL_DIR, f"{EXP_NAME}_multiclass_test_b0.15_0_0_0.csv")
    
    print("=" * 90)
    print(" COMPARATIVO DE PERFORMANCE — FOLD 0 (TEST SET BY-AUDIO) ")
    print("=" * 90)
    
    baseline = compute_metrics(baseline_csv)
    biased = compute_metrics(biased_csv)
    
    if baseline is None:
        print("Erro: CSV de baseline não encontrado!")
        return
    if biased is None:
        print("Erro: CSV de bias não encontrado!")
        return
        
    print(f"{'Métrica':<25} | {'Baseline (Sem Bias)':<22} | {'Calibrado (Bias Small=0.15)':<30} | {'Diferença':<12}")
    print("-" * 90)
    
    print(f"{'Índice SP (GM)':<25} | {baseline['sp_index']:20.2f}% | {biased['sp_index']:28.2f}% | {biased['sp_index'] - baseline['sp_index']:+10.2f}%")
    print(f"{'Acurácia Geral':<25} | {baseline['accuracy']:20.2f}% | {biased['accuracy']:28.2f}% | {biased['accuracy'] - baseline['accuracy']:+10.2f}%")
    
    print("-" * 90)
    print("Recalls por Classe:")
    for i in range(4):
        c_name = class_map[i]
        b_rec = baseline['recalls'][i]
        cal_rec = biased['recalls'][i]
        diff = cal_rec - b_rec
        print(f"  - {c_name:<21} | {b_rec:20.2f}% | {cal_rec:28.2f}% | {diff:+10.2f}%")
        
    print("=" * 90)
    print(" MATRIZ DE CONFUSÃO NORMALIZADA — BASELINE (SEM BIAS) ")
    print("-" * 90)
    print("True \\ Pred    | SMALL        | MEDIUM       | LARGE        | BACKGROUND  ")
    for r in range(4):
        cells = [f"{baseline['cm_norm'][r][c]:10.2f}%" for c in range(4)]
        print(f"{class_map[r]:<14} | " + " | ".join(cells))
        
    print("=" * 90)
    print(" MATRIZ DE CONFUSÃO NORMALIZADA — CALIBRADO (BIAS SMALL=0.15) ")
    print("-" * 90)
    print("True \\ Pred    | SMALL        | MEDIUM       | LARGE        | BACKGROUND  ")
    for r in range(4):
        cells = [f"{biased['cm_norm'][r][c]:10.2f}%" for c in range(4)]
        print(f"{class_map[r]:<14} | " + " | ".join(cells))
    print("=" * 90)

if __name__ == "__main__":
    main()
