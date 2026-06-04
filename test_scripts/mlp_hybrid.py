import enum
import argparse
import time
import sys
import os
import typing
# Add the project root to sys.path to allow imports from test_scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import pandas as pd
import numpy as np
import sklearn.decomposition as sk_decomp

import torch
import torch.utils.data as torch_data
import overrides

import iara.utils
import iara.ml.dataset as iara_dataset
import iara.default as iara_default
import iara.ml.models.mlp as iara_mlp
import iara.ml.experiment as iara_exp
import iara.ml.metrics as iara_metrics
import iara.ml.models.trainer as iara_trn
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc

from iara.default import DEFAULT_DIRECTORIES
from test_scripts.svm import HybridAudioFileProcessor

class PCADatasetWrapper(iara_dataset.BaseDataset):
    """Dataset wrapper that pre-computes PCA on the LOFAR portion of concatenated features to avoid on-the-fly bottleneck."""
    def __init__(self, original_dataset, pca):
        self.original_dataset = original_dataset
        self.pca = pca
        
        # Pre-compute transformed samples
        original_samples = original_dataset.get_samples().numpy()
        samples_mel = original_samples[:, :256]
        samples_lofar = original_samples[:, 256:]
        samples_lofar_pca = self.pca.transform(samples_lofar)
        samples_combined = np.hstack([samples_mel, samples_lofar_pca])
        self.sample_tensor = torch.tensor(samples_combined, dtype=torch.float32)
        
        # Targets
        self.target_tensor = original_dataset.get_targets().long()
        
    def __len__(self):
        return len(self.original_dataset)
        
    def __getitem__(self, idx) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return self.sample_tensor[idx], self.target_tensor[idx]

    def get_targets(self):
        return self.target_tensor

    def get_samples(self):
        return self.sample_tensor

    def get_file_samples(self, file_id: int):
        samples = []
        base_file_index = self.original_dataset.file_ids.index(file_id)
        limit_start = self.original_dataset.limit_ids[base_file_index]
        limit_end = self.original_dataset.limit_ids[base_file_index+1]
        
        for index in range(limit_start, limit_end):
            sample, target = self[index]
            samples.append(sample)
            
        return torch.stack(samples), target

    def get_file_ids(self):
        return self.original_dataset.get_file_ids()


class MLPHybridTrainer(iara_trn.OptimizerTrainer):
    """Custom OptimizerTrainer that applies PCA to LOFAR before neural network training/evaluation."""
    def __init__(self,
                 training_strategy: iara_trn.ModelTrainingStrategy,
                 trainer_id: str,
                 n_targets: int,
                 batch_size: int = 32,
                 n_epochs: int = 512,
                 patience: int = 16,
                 model_allocator = None,
                 optimizer_allocator = None,
                 loss_allocator = None,
                 n_pca_components: int = 64,
                 device: torch.device = iara.utils.get_available_device()) -> None:
        super().__init__(
            training_strategy=training_strategy,
            trainer_id=trainer_id,
            n_targets=n_targets,
            model_allocator=model_allocator,
            batch_size=batch_size,
            n_epochs=n_epochs,
            patience=patience,
            optimizer_allocator=optimizer_allocator,
            loss_allocator=loss_allocator,
            device=device
        )
        self.n_pca_components = n_pca_components
        self.pca = None

    def fit(self,
            model_base_dir: str,
            trn_dataset: iara_dataset.BaseDataset,
            val_dataset: iara_dataset.BaseDataset) -> None:
        if self.is_trained(model_base_dir=model_base_dir):
            return
            
        os.makedirs(model_base_dir, exist_ok=True)
        
        # Fit PCA on the LOFAR portion of training set
        samples = trn_dataset.get_samples().numpy()
        samples_lofar = samples[:, 256:]
        
        self.pca = sk_decomp.PCA(n_components=self.n_pca_components, random_state=42)
        self.pca.fit(samples_lofar)
        
        # Save the PCA object alongside the model
        pca_filename = os.path.join(model_base_dir, f"{self.trainer_id}_pca.pkl")
        with open(pca_filename, "wb") as f:
            pickle.dump(self.pca, f)
            
        # Wrap datasets and call super fit
        wrapped_trn = PCADatasetWrapper(trn_dataset, self.pca)
        wrapped_val = PCADatasetWrapper(val_dataset, self.pca)
        
        super().fit(model_base_dir, wrapped_trn, wrapped_val)

    def eval(self,
             eval_subset: iara_trn.Subset,
             eval_strategy: iara_trn.EvalStrategy,
             eval_base_dir: str,
             model_base_dir: typing.Optional[str] = None,
             dataset: iara_dataset.BaseDataset = None,
             complement_id: str = None) -> pd.DataFrame:
             
        # Check if the evaluation output file already exists
        output_file = self.output_filename(
            model_base_dir=eval_base_dir,
            complement=f'{str(eval_subset)}_{complement_id}' if complement_id is not None else str(eval_subset),
            extention='csv'
        )
        
        if os.path.exists(output_file):
            return super().eval(
                eval_subset=eval_subset,
                eval_strategy=eval_strategy,
                eval_base_dir=eval_base_dir,
                model_base_dir=model_base_dir,
                dataset=dataset,
                complement_id=complement_id
            )
            
        if model_base_dir is None:
            raise ValueError("model_base_dir must be provided if the evaluation output file does not exist.")
            
        # Load the saved PCA
        pca_filename = os.path.join(model_base_dir, f"{self.trainer_id}_pca.pkl")
        with open(pca_filename, "rb") as f:
            self.pca = pickle.load(f)
            
        if not isinstance(dataset, PCADatasetWrapper):
            wrapped_dataset = PCADatasetWrapper(dataset, self.pca)
        else:
            wrapped_dataset = dataset
            
        return super().eval(
            eval_subset=eval_subset,
            eval_strategy=eval_strategy,
            eval_base_dir=eval_base_dir,
            model_base_dir=model_base_dir,
            dataset=wrapped_dataset,
            complement_id=complement_id
        )


def main(folds: typing.List[int], n_pca_components: int = 64):
    output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/tests"
    directories = DEFAULT_DIRECTORIES

    grid = iara_metrics.GridCompiler()

    input_type = iara_dataset.InputType.Window()

    # Preprocessors for both spectrums
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
        name=f'mlp_hybrid_pca{n_pca_components}',
        dataset=iara_default.default_collection(),
        dataset_processor=dp_hybrid,
        output_base_dir=output_base_dir,
        input_type=input_type
    )

    # Input shape: 256 MEL + 64 LOFAR PCA = 320 features
    input_shape = [256 + n_pca_components]

    trainers = []
    trainers.append(MLPHybridTrainer(
        training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
        trainer_id=f'mlp_hybrid_pca{n_pca_components}',
        n_targets=config.dataset.target.get_n_targets(),
        batch_size=32,
        n_epochs=512,
        patience=16,
        n_pca_components=n_pca_components,
        model_allocator=lambda input_shape, n_targets,
            hidden_channels=[32, 16],
            dropout=0.2,
            norm_layer=torch.nn.BatchNorm1d,
            activation_layer=torch.nn.ReLU,
            activation_output_layer=torch.nn.Sigmoid:

                iara_mlp.MLP(
                    input_shape=input_shape,
                    hidden_channels=hidden_channels,
                    n_targets=n_targets,
                    dropout=dropout,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    activation_output_layer=activation_output_layer
                ),

        optimizer_allocator=lambda model,
            weight_decay=1e-3,
            lr=1e-4:
                torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr),
        loss_allocator=lambda class_weights:
                torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    ))

    manager = iara_exp.Manager(config, *trainers)

    result_grid = manager.run(folds=folds, override=False)

    for (eval_subset, eval_strategy), result_dict in result_grid.items():
        if eval_subset == iara_trn.Subset.ALL:
            continue

        for trainer_id, results in result_dict.items():
            for i_fold, result in enumerate(results):
                grid.add(params={
                    'eval_strategy': eval_strategy,
                    'eval_subset': eval_subset,
                },
                i_fold=i_fold,
                target=result['Target'],
                prediction=result['Prediction'])

    print(grid)


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description='RUN MLP Hybrid cross-validation')
    parser.add_argument('-F', '--fold', type=str, default=None,
                        help='Specify folds to be executed. Example: 0-9')
    parser.add_argument('--pca_components', type=int, default=64,
                        help='Number of PCA components for LOFAR. Default: 64')

    args = parser.parse_args()

    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(10)))

    main(folds=folds_to_execute, n_pca_components=args.pca_components)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {iara.utils.str_format_time(elapsed_time)}")
