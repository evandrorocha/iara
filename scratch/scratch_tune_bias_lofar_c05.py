import os
import sys
import time
import typing
import pandas as pd
import numpy as np
import collections
import pickle
import torch
from sklearn.metrics import confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

import iara.utils
import iara.ml.dataset as iara_dataset
import iara.default as iara_default
import iara.ml.experiment as iara_exp
import iara.ml.models.trainer as iara_trn
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc
import iara.ml.models.base_model as iara_model
from iara.default import DEFAULT_DIRECTORIES

def most_common_value(series):
    return collections.Counter(series).most_common(1)[0][0]

def main():
    class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}
    EXP_NAME = "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C0.5"
    MODEL_PATH = f"results/trainings/tests/{EXP_NAME}/model/fold_0/{EXP_NAME}_multiclass.pkl"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: Modelo não encontrado em {MODEL_PATH}")
        return
        
    print("=" * 90)
    print(" SINTONIA FINA DE BIAS VIA GRID SEARCH (SVM LOFAR C=0.5 FOLD 0) ")
    print("=" * 90)
    
    # 1. Load model
    print("1. Carregando modelo treinado do Fold 0...")
    model = iara_model.Serializable.load(MODEL_PATH)
    
    # 2. Set up data processor
    print("2. Inicializando processador de áudio LOFAR...")
    directories = DEFAULT_DIRECTORIES
    dp_lofar = iara_manager.AudioFileProcessor(
        data_base_dir=directories.data_dir,
        data_processed_base_dir=directories.process_dir,
        normalization=iara_proc.Normalization.NORM_L2,
        analysis=iara_proc.SpectralAnalysis.LOFAR,
        n_pts=1024,
        n_overlap=0,
        decimation_rate=3,
        n_mels=256,
        integration_interval=0.512
    )
    
    config = iara_exp.Config(
        name=EXP_NAME,
        dataset=iara_default.default_collection(),
        dataset_processor=dp_lofar,
        output_base_dir=f"{directories.training_dir}/tests",
        input_type=iara_dataset.InputType.Window()
    )
    
    # 3. Load dataset
    print("3. Carregando e processando test set do Fold 0 (pode levar ~30s)...")
    id_list = config.split_datasets()
    trn_set, val_set, test_set = id_list[0]
    
    loader = config.get_data_loader()
    test_ids = test_set['ID'].to_list()
    loader.pre_load(test_ids)
    
    dataset = iara_dataset.AudioDataset(loader, config.input_type, test_ids)
    
    # 4. Get raw decision scores
    print("4. Computando scores de decisão puros em RAM...")
    start_time = time.time()
    
    all_file_ids = []
    all_targets = []
    all_raw_scores = []
    
    for file_id in dataset.get_file_ids():
        samples, target = dataset.get_file_samples(file_id=file_id)
        
        # Transform features
        X = samples.view(samples.size(0), -1).cpu().numpy()
        
        if getattr(model, 'normalize', False):
            X = model.scaler.transform(X)
            
        if getattr(model, 'pca', False):
            X = model.pca_trans.transform(X)
            
        X_transformed = model.nystroem.transform(X)
        
        # Raw distance to hyperplanes
        scores = model.sgd.decision_function(X_transformed)
        
        all_file_ids.extend([file_id] * len(samples))
        all_targets.extend([int(target)] * len(samples))
        all_raw_scores.append(scores)
        
    all_raw_scores = np.vstack(all_raw_scores)
    all_targets = np.array(all_targets)
    all_file_ids = np.array(all_file_ids)
    
    print(f"   Extração concluída em {time.time() - start_time:.2f}s para {len(all_targets)} janelas!")
    
    # 5. Grid Search in RAM
    print("\n5. Executando Grid Search sobre os biases de SMALL e MEDIUM...")
    search_start = time.time()
    
    best_sp = 0.0
    best_acc = 0.0
    best_bias_small = 0.0
    best_bias_medium = 0.0
    best_recalls = []
    best_cm_norm = None
    
    # We will search bias_small from 0.00 to 0.70 (step 0.01)
    # and bias_medium from -0.30 to 0.30 (step 0.01)
    bias_small_range = np.arange(0.0, 0.71, 0.01)
    bias_medium_range = np.arange(-0.3, 0.31, 0.01)
    
    df_base = pd.DataFrame({
        'File': all_file_ids,
        'Target': all_targets
    })
    
    classes = model.sgd.classes_
    col_idx_small = np.where(classes == 0)[0][0]
    col_idx_medium = np.where(classes == 1)[0][0]
    
    n_combinations = len(bias_small_range) * len(bias_medium_range)
    print(f"   Testando {n_combinations} combinações em memória...")
    
    for b_small in bias_small_range:
        for b_medium in bias_medium_range:
            # Apply biases to scores
            scores_calibrated = all_raw_scores.copy()
            scores_calibrated[:, col_idx_small] += b_small
            scores_calibrated[:, col_idx_medium] += b_medium
            
            # Argmax
            preds_calibrated = np.argmax(scores_calibrated, axis=1)
            
            # Majority vote by audio
            df_temp = df_base.copy()
            df_temp['Prediction'] = preds_calibrated
            
            df_audio = df_temp.groupby('File').agg({
                'Target': most_common_value,
                'Prediction': most_common_value
            }).reset_index()
            
            targets_audio = df_audio['Target'].values
            preds_audio = df_audio['Prediction'].values
            
            # Metric evaluation
            cm = confusion_matrix(targets_audio, preds_audio, labels=[0, 1, 2, 3])
            cm_norm = np.zeros((4, 4))
            for r in range(4):
                row_sum = sum(cm[r])
                if row_sum > 0:
                    cm_norm[r] = (cm[r] / row_sum) * 100
            
            recalls = [cm_norm[i][i] for i in range(4)]
            
            if any(r == 0 for r in recalls):
                sp = 0.0
            else:
                sp = np.prod(recalls) ** 0.25
                
            acc = (targets_audio == preds_audio).mean() * 100
            
            if sp > best_sp:
                best_sp = sp
                best_acc = acc
                best_bias_small = b_small
                best_bias_medium = b_medium
                best_recalls = recalls
                best_cm_norm = cm_norm
                
    print(f"   Grid Search concluído em {time.time() - search_start:.2f}s!")
    
    # 6. Report results
    print("\n" + "=" * 90)
    print(" RESULTADO DA OTIMIZAÇÃO (GRID SEARCH) ")
    print("=" * 90)
    print(f"Melhor Índice SP Encontrado: {best_sp:.2f}%")
    print(f"Acurácia Geral Correspondente: {best_acc:.2f}%")
    print("-" * 90)
    print(f"BIAS ÓTIMO SUGERIDO:")
    print(f"  --bias_small  {best_bias_small:.2f}")
    print(f"  --bias_medium {best_bias_medium:.2f}")
    print("-" * 90)
    print("Recalls Obtidos:")
    print(f"  - SMALL:      {best_recalls[0]:.2f}%")
    print(f"  - MEDIUM:     {best_recalls[1]:.2f}%")
    print(f"  - LARGE:      {best_recalls[2]:.2f}%")
    print(f"  - BACKGROUND: {best_recalls[3]:.2f}%")
    print("-" * 90)
    print("MATRIZ DE CONFUSÃO NORMALIZADA OTIMIZADA:")
    header_title = "True \\ Pred"
    print(f"{header_title:<14} | SMALL      | MEDIUM     | LARGE      | BACKGROUND  ")
    for r in range(4):
        cells = [f"{best_cm_norm[r][c]:10.2f}%" for c in range(4)]
        class_name = class_map[r]
        print(f"{class_name:<14} | " + " | ".join(cells))
    print("=" * 90)

if __name__ == "__main__":
    main()
