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

# Class definitions to allow pickle deserialization in __main__ namespace
class SVMNystroemHybrid(iara_model.BaseModel):
    def __init__(self,
                 n_components: int = 300,
                 gamma: typing.Union[str, float] = 'scale',
                 C: float = 1.0,
                 n_targets: int = 4,
                 n_mel_features: int = 256,
                 n_pca_components: int = 64,
                 normalize: bool = True,
                 penalty: str = 'l2',
                 l1_ratio: float = 0.15,
                 biases: typing.Optional[typing.List[float]] = None,
                 random_state: int = 42):
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma
        self.C = C
        self.n_targets = n_targets
        self.n_mel_features = n_mel_features
        self.n_pca_components = n_pca_components
        self.normalize = normalize
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.biases = biases
        self.random_state = random_state
        self.pca_trans = PCA(n_components=n_pca_components, random_state=random_state)
        self.scaler = StandardScaler() if normalize else None
        self.nystroem = Nystroem(kernel='rbf', gamma=None, n_components=n_components, random_state=random_state)
        self.sgd = SGDClassifier(loss='hinge', penalty=penalty, l1_ratio=l1_ratio, alpha=1.0, class_weight='balanced', max_iter=1000, tol=1e-3, random_state=random_state, n_jobs=1)
        self.is_fitted = False

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling forward().")
            
        X = data.view(data.size(0), -1).cpu().numpy()
        X_mel = X[:, :self.n_mel_features]
        X_lofar = X[:, self.n_mel_features:]
        
        X_lofar_pca = self.pca_trans.transform(X_lofar)
        X_combined = np.hstack([X_mel, X_lofar_pca])
        
        if self.normalize:
            X_combined = self.scaler.transform(X_combined)
            
        X_transformed = self.nystroem.transform(X_combined)
        
        scores = self.sgd.decision_function(X_transformed)
        biases = getattr(self, 'biases', None)
        if biases is not None:
            if len(scores.shape) == 1 or scores.shape[1] == 1:
                bias_val = biases[1] - biases[0]
                if len(scores.shape) == 1:
                    scores += bias_val
                    predictions = (scores > 0).astype(int)
                else:
                    scores[:, 0] += bias_val
                    predictions = (scores[:, 0] > 0).astype(int)
            else:
                for i, cls in enumerate(self.sgd.classes_):
                    if cls < len(biases):
                        scores[:, i] += biases[cls]
                predictions = np.argmax(scores, axis=1)
        else:
            predictions = self.sgd.predict(X_transformed)
            
        return torch.tensor(predictions, dtype=torch.long)

class HybridAudioFileProcessor(iara_manager.AudioFileProcessor):
    def __init__(self, mel_processor, lofar_processor):
        self.mel_processor = mel_processor
        self.lofar_processor = lofar_processor
        self.data_base_dir = mel_processor.data_base_dir
        self.data_processed_base_dir = mel_processor.data_processed_base_dir
        self.normalization = mel_processor.normalization
        self.analysis = "hybrid"
        self.frequency_limit = None
        self.integration_overlap = mel_processor.integration_overlap
        self.integration_interval = mel_processor.integration_interval
        self.n_pts = mel_processor.n_pts
        self.n_overlap = mel_processor.n_overlap
        self.n_mels = mel_processor.n_mels
        self.decimation_rate = mel_processor.decimation_rate
        self._check_dir()

    def _get_hash(self) -> str:
        return "hybrid_fusion_mel_lofar_v1"

    def get_data(self, file_id: int) -> typing.Tuple[pd.DataFrame, np.ndarray]:
        df_mel, times = self.mel_processor.get_data(file_id)
        df_lofar, _ = self.lofar_processor.get_data(file_id)
        df_mel_renamed = df_mel.copy()
        df_mel_renamed.columns = [f'mel_{i}' for i in range(df_mel.shape[1])]
        df_lofar_renamed = df_lofar.copy()
        df_lofar_renamed.columns = [f'lofar_{i}' for i in range(df_lofar.shape[1])]
        df_hybrid = pd.concat([df_mel_renamed, df_lofar_renamed], axis=1)
        return df_hybrid, times

def main():
    EXP_NAME = "svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5"
    base_dir = f"results/trainings/tests/{EXP_NAME}"
    biases = [0.40, -0.19, 0.0, 0.0]
    
    print("=" * 90)
    print(" FAST CALIBRATED GENERATOR — GENERATING FOLDS 2-9 ")
    print("=" * 90)
    
    # Initialize processors
    directories = DEFAULT_DIRECTORIES
    dp_mel = iara_manager.AudioFileProcessor(
        data_base_dir=directories.data_dir,
        data_processed_base_dir=directories.process_dir,
        normalization=iara_proc.Normalization.NORM_L2,
        analysis=iara_proc.SpectralAnalysis.LOG_MELGRAM,
        n_pts=1024,
        n_overlap=0,
        decimation_rate=3,
        n_mels=256,
        integration_interval=0.512
    )
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
    dp_hybrid = HybridAudioFileProcessor(dp_mel, dp_lofar)
    
    config = iara_exp.Config(
        name=EXP_NAME,
        dataset=iara_default.default_collection(),
        dataset_processor=dp_hybrid,
        output_base_dir=f"{directories.training_dir}/tests",
        input_type=iara_dataset.InputType.Window()
    )
    
    # Load splits
    id_list = config.split_datasets()
    loader = config.get_data_loader()
    
    # Generate calibrated test CSV for folds 2 to 9
    for fold in range(2, 10):
        print(f"\n---> Processing Fold {fold}...")
        model_path = os.path.join(base_dir, "model", f"fold_{fold}", f"{EXP_NAME}_multiclass.pkl")
        output_path = os.path.join(base_dir, "eval", f"fold_{fold}", f"{EXP_NAME}_multiclass_test_b0.4_-0.19_0_0.csv")
        
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
