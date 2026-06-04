import os
import sys
import collections
import pandas as pd
import numpy as np
import pickle
import torch
import time
import typing

# Set standard output encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier

import iara.utils
import iara.ml.dataset as iara_dataset
import iara.default as iara_default
import iara.ml.experiment as iara_exp
import iara.ml.models.trainer as iara_trn
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc
import iara.ml.models.base_model as iara_model
from iara.default import DEFAULT_DIRECTORIES

# Class definitions to allow pickle deserialization of SVMNystroemPreprocessed in main
# (required on some systems depending on sys.path namespaces)

def main():
    EXP_NAME = "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C0.5"
    base_dir = f"results/trainings/tests/{EXP_NAME}"
    biases = [0.32, 0.27, 0.0, 0.0]
    
    print("=" * 90)
    print(" FAST CALIBRATED GENERATOR — LOFAR C=0.5 (FOLDS 0-9) ")
    print("=" * 90)
    
    # Initialize processors
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
    
    # Load splits
    id_list = config.split_datasets()
    loader = config.get_data_loader()
    
    # Generate calibrated test CSV for folds 0 to 9
    for fold in range(10):
        print(f"\n---> Processing Fold {fold}...")
        model_path = os.path.join(base_dir, "model", f"fold_{fold}", f"{EXP_NAME}_multiclass.pkl")
        output_path = os.path.join(base_dir, "eval", f"fold_{fold}", f"{EXP_NAME}_multiclass_test_b0.32_0.27_0_0.csv")
        
        if os.path.exists(output_path):
            print(f"     Output already exists: {output_path}. Skipping.")
            continue
            
        if not os.path.exists(model_path):
            print(f"     Error: Model pickle not found at {model_path}. Skipping.")
            continue
            
        # Load model and inject biases
        model = iara_model.Serializable.load(model_path)
        model.biases = biases
        
        # Load test subset
        _, _, test_set = id_list[fold]
        test_ids = test_set['ID'].to_list()
        
        print(f"     Preloading test set ({len(test_ids)} audios)...")
        loader.pre_load(test_ids)
        dataset = iara_dataset.AudioDataset(loader, config.input_type, test_ids)
        
        records = []
        start_time = time.time()
        
        # Run inference
        for file_id in dataset.get_file_ids():
            samples, target = dataset.get_file_samples(file_id=file_id)
            preds = model.forward(samples).cpu().numpy()
            for p in preds:
                records.append({
                    'File': file_id,
                    'Target': int(target),
                    'Prediction': int(p)
                })
                
        df_out = pd.DataFrame(records)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_out.to_csv(output_path, index=False)
        
        print(f"     Saved calibrated predictions to {output_path}")
        print(f"     Completed Fold {fold} in {time.time() - start_time:.2f} seconds!")
        
    print("\nGeneration finished successfully!")

if __name__ == "__main__":
    main()
