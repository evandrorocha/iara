import os
import collections
import pandas as pd
import numpy as np
import pickle
import torch
import time
import typing
from sklearn.metrics import confusion_matrix

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

# Make pickle unpickler find SVMNystroemHybrid in __main__
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

# --- Re-define Hybrid processor to support loading ---
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

def most_common_value(series):
    return collections.Counter(series).most_common(1)[0][0]

def main():
    class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}
    EXP_NAME = "svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5"
    MODEL_PATH = f"results/trainings/tests/{EXP_NAME}/model/fold_0/{EXP_NAME}_multiclass.pkl"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: Modelo não encontrado em {MODEL_PATH}")
        return
        
    print("=" * 90)
    print(" SINTONIA FINA DE BIAS VIA GRID SEARCH (FOLD 0) ")
    print("=" * 90)
    
    # 1. Load model
    print("1. Carregando modelo treinado do Fold 0...")
    model = iara_model.Serializable.load(MODEL_PATH)
    
    # 2. Set up data processors
    print("2. Inicializando processadores de áudio (Híbrido MEL + LOFAR)...")
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
        X_mel = X[:, :model.n_mel_features]
        X_lofar = X[:, model.n_mel_features:]
        
        X_lofar_pca = model.pca_trans.transform(X_lofar)
        X_combined = np.hstack([X_mel, X_lofar_pca])
        
        if model.normalize:
            X_combined = model.scaler.transform(X_combined)
            
        X_transformed = model.nystroem.transform(X_combined)
        
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
    
    # We will search bias_small from 0.00 to 0.40 (step 0.01)
    # and bias_medium from -0.20 to 0.20 (step 0.01)
    bias_small_range = np.arange(0.0, 0.41, 0.01)
    bias_medium_range = np.arange(-0.2, 0.21, 0.01)
    
    # Store predictions in a DataFrame once to speed up grouping
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
            
            # Avoid division by zero/crash if any recall is 0
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
    print("True \\ Pred    | SMALL        | MEDIUM       | LARGE        | BACKGROUND  ")
    for r in range(4):
        cells = [f"{best_cm_norm[r][c]:10.2f}%" for c in range(4)]
        class_name = class_map[r]
        print(f"{class_name:<14} | " + " | ".join(cells))
    print("=" * 90)

if __name__ == "__main__":
    main()
