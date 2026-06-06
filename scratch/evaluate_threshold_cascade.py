import os
import sys
import time
import typing
import pandas as pd
import numpy as np
import collections
import torch

sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('.'))

# Fix pickle loading from test_scripts/svm.py classes
import __main__
sys.path.append(os.path.abspath('test_scripts'))
import svm as svm_module
__main__.SVMNystroemHybridCascaded = svm_module.SVMNystroemHybridCascaded
__main__.SVMNystroemLocal = svm_module.SVMNystroemLocal
__main__.SVMNystroemCascaded = svm_module.SVMNystroemCascaded
__main__.HybridAudioFileProcessor = svm_module.HybridAudioFileProcessor


import iara.utils
import iara.ml.dataset as iara_dataset
import iara.default as iara_default
import iara.ml.experiment as iara_exp
import iara.ml.metrics as iara_metrics
import iara.ml.models.trainer as iara_trn
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc
import iara.ml.models.base_model as iara_model

from iara.default import DEFAULT_DIRECTORIES
from test_scripts.svm import HybridAudioFileProcessor

def run_evaluation(threshold: float, model, test_dataset):
    samples = test_dataset.get_samples()
    # model is a SVMNystroemHybridCascaded object
    
    # 1. Get predictions from multiclass model on MEL features
    data_mel = samples[:, :256]
    preds_win_tensor = model.multiclass_model.forward(data_mel)
    preds_win = preds_win_tensor.cpu().numpy()
    
    # 2. Re-evaluate MEDIUM (1) predictions
    medium_mask = (preds_win == 1)
    if medium_mask.any():
        data_lofar_med = samples[medium_mask, 256:]
        X = data_lofar_med.view(data_lofar_med.size(0), -1).cpu().numpy()
        
        # Apply specialist normalize/scaler
        if getattr(model.specialist_model, 'normalize', False) and getattr(model.specialist_model, 'scaler', None) is not None:
            X = model.specialist_model.scaler.transform(X)
        # Apply specialist pca
        if getattr(model.specialist_model, 'pca', False) and getattr(model.specialist_model, 'pca_trans', None) is not None:
            X = model.specialist_model.pca_trans.transform(X)
        # Apply Nystroem
        X_transformed = model.specialist_model.nystroem.transform(X)
        
        if getattr(model.specialist_model, 'rms', False):
            _, rms = model.specialist_model._normalize_and_extract_rms(data_lofar_med.view(data_lofar_med.size(0), -1).cpu().numpy())
            rms_scaled = model.specialist_model.rms_scaler.transform(rms) * model.specialist_model.rms_scale
            X_final = np.hstack([X_transformed, rms_scaled])
        else:
            X_final = X_transformed
            
        scores = model.specialist_model.sgd.decision_function(X_final)
        
        # Specialist multiclass (scores shape (n, 4))
        c0, c1 = 0, 1
        bias_c0 = model.cascade_bias_small if c0 == 0 else 0.0
        bias_c1 = model.cascade_bias_small if c1 == 0 else 0.0
        
        score_c0 = scores[:, c0] + bias_c0
        score_c1 = scores[:, c1] + bias_c1
        
        # Softmax confidence
        max_score = np.maximum(score_c0, score_c1)
        exp_c0 = np.exp(score_c0 - max_score)
        exp_c1 = np.exp(score_c1 - max_score)
        p_c0 = exp_c0 / (exp_c0 + exp_c1)
        p_c1 = exp_c1 / (exp_c0 + exp_c1)
        conf = np.maximum(p_c0, p_c1)
        
        spec_preds = np.where(score_c0 > score_c1, c0, c1)
        
        # Revert to original (1/MEDIUM) if conf < threshold
        trust_mask = (conf >= threshold)
        final_spec_preds = np.where(trust_mask, spec_preds, 1)
        preds_win[medium_mask] = final_spec_preds

    # 3. Re-evaluate LARGE (2) predictions
    large_mask = (preds_win == 2)
    if large_mask.any():
        data_lofar_large = samples[large_mask, 256:]
        X = data_lofar_large.view(data_lofar_large.size(0), -1).cpu().numpy()
        
        # Apply specialist normalize/scaler
        if getattr(model.specialist_model, 'normalize', False) and getattr(model.specialist_model, 'scaler', None) is not None:
            X = model.specialist_model.scaler.transform(X)
        # Apply specialist pca
        if getattr(model.specialist_model, 'pca', False) and getattr(model.specialist_model, 'pca_trans', None) is not None:
            X = model.specialist_model.pca_trans.transform(X)
        # Apply Nystroem
        X_transformed = model.specialist_model.nystroem.transform(X)
        
        if getattr(model.specialist_model, 'rms', False):
            _, rms = model.specialist_model._normalize_and_extract_rms(data_lofar_large.view(data_lofar_large.size(0), -1).cpu().numpy())
            rms_scaled = model.specialist_model.rms_scaler.transform(rms) * model.specialist_model.rms_scale
            X_final = np.hstack([X_transformed, rms_scaled])
        else:
            X_final = X_transformed
            
        scores = model.specialist_model.sgd.decision_function(X_final)
        
        c0, c2 = 0, 2
        bias_c0 = model.cascade_bias_small if c0 == 0 else 0.0
        bias_c2 = model.cascade_bias_small if c2 == 0 else 0.0
        
        score_c0 = scores[:, c0] + bias_c0
        score_c2 = scores[:, c2] + bias_c2
        
        # Softmax confidence
        max_score = np.maximum(score_c0, score_c2)
        exp_c0 = np.exp(score_c0 - max_score)
        exp_c2 = np.exp(score_c2 - max_score)
        p_c0 = exp_c0 / (exp_c0 + exp_c2)
        p_c2 = exp_c2 / (exp_c0 + exp_c2)
        conf = np.maximum(p_c0, p_c2)
        
        spec_preds = np.where(score_c0 > score_c2, c0, c2)
        
        # Revert to original (2/LARGE) if conf < threshold
        trust_mask = (conf >= threshold)
        final_spec_preds = np.where(trust_mask, spec_preds, 2)
        preds_win[large_mask] = final_spec_preds
        
    # 4. Audio-level Evaluation
    limit_ids = test_dataset.limit_ids if hasattr(test_dataset, 'limit_ids') else test_dataset.original_dataset.limit_ids
    file_ids_list = test_dataset.file_ids if hasattr(test_dataset, 'file_ids') else test_dataset.original_dataset.file_ids
    
    file_ids = []
    all_targets = []
    
    for file_id in test_dataset.get_file_ids():
        base_file_index = file_ids_list.index(file_id)
        n_samples = limit_ids[base_file_index+1] - limit_ids[base_file_index]
        target = test_dataset.loader.target_map[file_id] if hasattr(test_dataset, 'loader') else test_dataset.original_dataset.loader.target_map[file_id]
        file_ids.extend([file_id] * n_samples)
        all_targets.extend([int(target)] * n_samples)
        
    df = pd.DataFrame({"File": file_ids, "Target": all_targets, "Prediction": preds_win})
    
    def most_common_value(series):
        return collections.Counter(series).most_common(1)[0][0]
        
    df_audio = df.groupby('File').agg({
        'Target': most_common_value,
        'Prediction': most_common_value
    }).reset_index()
    
    y_true = df_audio['Target'].values
    y_pred = df_audio['Prediction'].values
    
    acc = iara_metrics.Metric.ACCURACY.compute(y_true, y_pred)
    sp = iara_metrics.Metric.SP_INDEX.compute(y_true, y_pred)
    
    # Recalls
    from sklearn.metrics import recall_score
    recalls = recall_score(y_true, y_pred, average=None)
    
    return acc, sp, recalls

def main():
    print("Initializing dataset and processor...")
    norm_mode = iara_proc.Normalization.NORM_L2
    
    dp_mel = iara_manager.AudioFileProcessor(
        data_base_dir=DEFAULT_DIRECTORIES.data_dir,
        data_processed_base_dir=DEFAULT_DIRECTORIES.process_dir,
        normalization=norm_mode,
        analysis=iara_proc.SpectralAnalysis.LOG_MELGRAM,
        n_pts=1024,
        n_overlap=0,
        decimation_rate=3,
        n_mels=256,
        integration_interval=0.512
    )
    dp_lofar = iara_manager.AudioFileProcessor(
        data_base_dir=DEFAULT_DIRECTORIES.data_dir,
        data_processed_base_dir=DEFAULT_DIRECTORIES.process_dir,
        normalization=norm_mode,
        analysis=iara_proc.SpectralAnalysis.LOFAR,
        n_pts=1024,
        n_overlap=0,
        decimation_rate=3,
        n_mels=256,
        integration_interval=0.512
    )
    dp = HybridAudioFileProcessor(dp_mel, dp_lofar)
    
    config = iara_exp.Config(
        name="temp_config",
        dataset=iara_default.default_collection(),
        dataset_processor=dp,
        output_base_dir="results/trainings/tests",
        input_type=iara_dataset.InputType.Window()
    )
    
    id_list = config.split_datasets()
    (trn_set, val_set, test_set) = id_list[0] # fold 0
    
    test_dataset = iara_dataset.AudioDataset(
        config.get_data_loader(),
        config.input_type,
        test_set['ID'].to_list()
    )
    
    print("Loading hybrid cascade model...")
    model_path = "results/trainings/tests/svm_nystroem_4000_hybrid_cascade_pretrained_svm_nystroem_6000_lofar_elasticnet_l1r0.15_elasticnet_l1r0.15/model/fold_0/svm_nystroem_4000_hybrid_cascade_pretrained_svm_nystroem_6000_lofar_elasticnet_l1r0.15_elasticnet_l1r0.15_multiclass.pkl"
    model = iara_model.BaseModel.load(model_path)
    model.cascade_bias_small = 0.0 # reset bias or use default
    
    print("Evaluating thresholds...")
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    
    results = []
    for t in thresholds:
        acc, sp, recalls = run_evaluation(t, model, test_dataset)
        results.append({
            "threshold": t,
            "acc": acc,
            "sp": sp,
            "r_small": recalls[0]*100.0,
            "r_medium": recalls[1]*100.0,
            "r_large": recalls[2]*100.0,
            "r_bg": recalls[3]*100.0
        })
        print(f"t={t:.2f}: ACC={acc:.2f}%, SP={sp:.2f}%, SMALL={recalls[0]*100.0:.2f}%, MEDIUM={recalls[1]*100.0:.2f}%, LARGE={recalls[2]*100.0:.2f}%, BG={recalls[3]*100.0:.2f}%")
        
    df_results = pd.DataFrame(results)
    df_results.to_csv("scratch/threshold_results.csv", index=False)
    print("\nResults saved to scratch/threshold_results.csv")

if __name__ == "__main__":
    main()
