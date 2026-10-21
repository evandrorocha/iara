"""
Train a binary SMALL vs LARGE SVM specialist on hybrid MEL+LOFAR features.

The specialist is trained ONLY on SMALL (class 0) and LARGE (class 2) samples,
using concatenated MEL (256) + LOFAR-PCA (k) features. LARGE ships have very
distinctive LOFAR tonals (propeller harmonics), so fewer PCA components are
needed compared to the SMALL vs MEDIUM specialist (PCA4-16 vs PCA64).

LARGE (class 2) is remapped to label 1 internally so the model outputs [0,1].
The cascade's _evaluate_specialist maps back using allowed_classes=[0, 2].

Usage (inside Docker container):
    # Train all 10 folds (default PCA4):
    python test_scripts/svm_binary_specialist_large.py -F 0-9

    # Train with PCA8 for comparison:
    python test_scripts/svm_binary_specialist_large.py -F 0-9 --pca_components 8

    # Then use as pre-trained large specialist in the hybrid cascade:
    python test_scripts/svm.py --cascade --components 4000 \\
        --pretrained_specialist_dir results/trainings/tests/svm_nystroem_6000_hybrid_binary_small_medium_pca64_elasticnet_l1r0.15 \\
        --pretrained_large_specialist_dir results/trainings/tests/svm_nystroem_6000_hybrid_binary_small_large_pca4_elasticnet_l1r0.15 \\
        -F 0-9
"""
import time
import typing
import argparse
import itertools
import os
import sys
import types

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for _mod_name in ['overrides', 'tikzplotlib', 'librosa']:
    try:
        __import__(_mod_name)
    except ImportError:
        _m = types.ModuleType(_mod_name)
        if _mod_name == 'overrides':
            _m.override = lambda f: f
        sys.modules[_mod_name] = _m

import iara.utils
import iara.records
import iara.ml.dataset as iara_dataset
import iara.default as iara_default
import iara.ml.experiment as iara_exp
import iara.ml.metrics as iara_metrics
import iara.ml.models.trainer as iara_trn
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc

from iara.default import DEFAULT_DIRECTORIES
from svm import SVMNystroemHybridTrainer, HybridAudioFileProcessor


class SmallLargeFilter:
    """Keep only SMALL (length < 50m) and LARGE (length >= 100m) ships.
    Drops MEDIUM (50 <= length < 100m) and BACKGROUND (non-numeric length).
    Applied before the target function so GenericTarget sees only labels 0 and 1.
    """
    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        lengths = pd.to_numeric(input_df['Length'], errors='coerce')
        return input_df[(lengths < 50) | (lengths >= 100)]


def classify_row_small_large(row) -> int:
    """Map SMALL→0, LARGE→1 (remapped from 2). MEDIUM/BG are pre-filtered out."""
    base = iara_default.Target.classify_row(row)
    if base == 2:   # LARGE → remap to label 1
        return 1
    return base     # 0 for SMALL


def filter_4class_split_to_small_large(split_df: pd.DataFrame) -> pd.DataFrame:
    """Filter a 4-class split DataFrame to keep only SMALL (0) and LARGE (2),
    remapping LARGE (2) -> 1 for binary classification."""
    filtered = split_df[split_df['Target'].isin([0, 2])].copy()
    filtered['Target'] = filtered['Target'].map({0: 0, 2: 1}).astype(int)
    return filtered


class AlignedSmallLargeConfig(iara_exp.Config):
    """Config that computes 5x2cv splits on the full 4-class default_collection() first,
    then filters each fold's (trn_set, val_set, test_set) to SMALL (0) and LARGE (2 -> 1).
    Guarantees 0% test leakage relative to the 4-class splits.
    """
    def split_datasets(self) -> typing.List[typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        full_config = iara_exp.Config(
            name=self.name,
            dataset=iara_default.default_collection(),
            dataset_processor=self.dataset_processor,
            output_base_dir=os.path.dirname(self.output_base_dir),
            input_type=self.input_type,
            test_ratio=self.test_ratio,
            exclusive_ships_on_test=self.exclusive_ships_on_test,
            exclusive_header=self.exclusive_header,
            target_header=self.target_header,
        )
        full_splits = full_config.split_datasets()
        return [
            (
                filter_4class_split_to_small_large(trn_4c),
                filter_4class_split_to_small_large(val_4c),
                filter_4class_split_to_small_large(test_4c),
            )
            for trn_4c, val_4c, test_4c in full_splits
        ]


def main(
    folds: typing.List[int],
    n_components: int = 6000,
    C: float = 1.0,
    gamma: typing.Union[str, float] = 'scale',
    n_pca_components: int = 4,
    penalty: str = 'elasticnet',
    l1_ratio: float = 0.15,
    class_weight: typing.Union[str, None] = 'balanced',
    override: bool = False,
):
    output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/tests"
    directories = DEFAULT_DIRECTORIES

    grid = iara_metrics.GridCompiler()
    input_type = iara_dataset.InputType.Window()

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
    dp = HybridAudioFileProcessor(dp_mel, dp_lofar)

    # SmallLargeFilter drops MEDIUM and BG; classify_row_small_large remaps LARGE→1
    # so GenericTarget(n_targets=2) sees only labels [0, 1] and keeps both.
    binary_collection = iara.records.CustomCollection(
        collection=iara.records.Collection.OS,
        target=iara.records.GenericTarget(
            n_targets=2,
            function=classify_row_small_large,
            include_others=False
        ),
        filters=SmallLargeFilter(),
        only_sample=False
    )

    name_parts = [f'svm_nystroem_{n_components}', 'hybrid', 'binary_small_large',
                  f'pca{n_pca_components}']
    if penalty != 'l2':
        name_parts.append(penalty)
        if penalty == 'elasticnet':
            name_parts.append(f'l1r{l1_ratio}')
    if C != 1.0:
        name_parts.append(f'C{C}')
    exp_name = '_'.join(name_parts)

    # 1. Split on full 4-class default_collection() FIRST to align fold_0..fold_9
    full_4class_config = iara_exp.Config(
        name=exp_name,
        dataset=iara_default.default_collection(),
        dataset_processor=dp,
        output_base_dir=output_base_dir,
        input_type=input_type
    )
    full_4class_splits = full_4class_config.split_datasets()

    config = AlignedSmallLargeConfig(
        name=exp_name,
        dataset=binary_collection,
        dataset_processor=dp,
        output_base_dir=output_base_dir,
        input_type=input_type
    )

    trainer = SVMNystroemHybridTrainer(
        training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
        trainer_id=exp_name,
        n_targets=2,
        n_components=n_components,
        gamma=gamma,
        C=C,
        n_mel_features=256,
        n_pca_components=n_pca_components,
        normalize=True,
        penalty=penalty,
        l1_ratio=l1_ratio,
        biases=None,
        class_weight=class_weight,
    )

    manager = iara_exp.Manager(config, trainer)
    if override and os.path.exists(config.output_base_dir):
        iara.utils.backup_folder(base_dir=config.output_base_dir,
                                 time_str_format=iara_exp.Config.TIME_STR_FORMAT)
    os.makedirs(config.output_base_dir, exist_ok=True)
    config_file = os.path.join(config.output_base_dir, f"{config.name}.json")
    if override or not os.path.exists(config_file):
        try:
            config.save(config.output_base_dir)
        except OSError:
            pass

    eval_subsets = [iara_trn.Subset.TRN, iara_trn.Subset.VAL, iara_trn.Subset.TEST]

    for i_fold in folds:
        trn_4c, val_4c, test_4c = full_4class_splits[i_fold]
        trn_set = filter_4class_split_to_small_large(trn_4c)
        val_set = filter_4class_split_to_small_large(val_4c)
        test_set = filter_4class_split_to_small_large(test_4c)

        # Strict zero-leakage verification against 4-class test set
        overlap = set(trn_set['ID']) & set(test_4c['ID'])
        assert len(overlap) == 0, f"Leakage detected in fold {i_fold}: {len(overlap)} overlapping IDs!"

        print(f"--- Fold {i_fold}: trn={len(trn_set)} (SMALL={(trn_set['Target']==0).sum()}, LARGE={(trn_set['Target']==1).sum()}), "
              f"val={len(val_set)}, test={len(test_set)} [0% overlap with 4-class test_set] ---")

        manager.fit(
            i_fold=i_fold,
            trn_dataset_ids=trn_set['ID'].to_list(),
            val_dataset_ids=val_set['ID'].to_list()
        )

        id_set = {
            iara_trn.Subset.TRN: trn_set,
            iara_trn.Subset.VAL: val_set,
            iara_trn.Subset.TEST: test_set,
        }
        for eval_subset, eval_strategy in itertools.product(eval_subsets, iara_trn.EvalStrategy):
            manager.eval(
                i_fold=i_fold,
                eval_subset=eval_subset,
                eval_strategy=eval_strategy,
                dataset_ids=id_set[eval_subset]['ID'].to_list()
            )

    for eval_subset, eval_strategy in itertools.product(eval_subsets, iara_trn.EvalStrategy):
        result_dict = manager.compile_results(
            eval_subset=eval_subset,
            eval_strategy=eval_strategy,
            trainer_list=manager.trainer_list,
            folds=folds
        )
        for trainer_id, results in result_dict.items():
            for idx, result in enumerate(results):
                grid.add(
                    params={
                        'eval_strategy': eval_strategy,
                        'eval_subset': eval_subset,
                    },
                    i_fold=folds[idx],
                    target=result['Target'],
                    prediction=result['Prediction']
                )

    print(grid)

    spec_dir = os.path.join(output_base_dir, exp_name)
    medium_spec_dir = os.path.join(output_base_dir,
                                   'svm_nystroem_6000_hybrid_binary_small_medium_pca64_elasticnet_l1r0.15')
    print(f"\n[Done] Binary hybrid SMALL vs LARGE specialist saved to: {spec_dir}")
    print(f"\n[Cascade] To use this specialist in the hybrid cascade:")
    print(f"  python test_scripts/svm.py --cascade --components 4000 \\")
    print(f"    --pretrained_specialist_dir {medium_spec_dir} \\")
    print(f"    --pretrained_large_specialist_dir {spec_dir} \\")
    print(f"    -F 0-9")


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='Train binary SMALL vs LARGE SVM specialist on hybrid MEL+LOFAR features.'
    )
    parser.add_argument(
        '-F', '--fold', type=str, default='0',
        help='Folds to run (e.g. 0, 0-9, 0,2,4). Default: 0'
    )
    parser.add_argument(
        '--components', type=int, default=6000,
        help='Number of Nyström components. Default: 6000'
    )
    parser.add_argument(
        '--reg_c', type=float, default=1.0,
        help='Regularization C for SGDClassifier. Default: 1.0'
    )
    parser.add_argument(
        '--gamma', type=str, default='scale',
        help='Nyström gamma (auto, scale, or a float). Default: scale'
    )
    parser.add_argument(
        '--pca_components', type=int, default=4,
        help='Number of PCA components applied to LOFAR portion. Default: 4'
    )
    parser.add_argument(
        '--penalty', type=str, default='elasticnet',
        choices=['l2', 'l1', 'elasticnet'],
        help='SGD regularization penalty. Default: elasticnet'
    )
    parser.add_argument(
        '--l1_ratio', type=float, default=0.15,
        help='ElasticNet L1 ratio. Default: 0.15'
    )
    parser.add_argument(
        '--no_balanced', action='store_true', default=False,
        help='Disable balanced class weighting. Balanced is the default.'
    )
    parser.add_argument(
        '--override', action='store_true', default=False,
        help='Backup existing experiment folder and retrain from scratch.'
    )

    args = parser.parse_args()

    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(1)))

    gamma_val = args.gamma
    try:
        gamma_val = float(args.gamma)
    except ValueError:
        pass

    class_weight = None if args.no_balanced else 'balanced'

    print(f"[Binary Specialist] SMALL vs LARGE on HYBRID (MEL+LOFAR) features")
    print(f"  folds={folds_to_execute}")
    print(f"  n_components={args.components}, C={args.reg_c}, gamma={gamma_val}")
    print(f"  pca_components={args.pca_components} (applied to LOFAR only; MEL=256 kept as-is)")
    print(f"  penalty={args.penalty}, l1_ratio={args.l1_ratio}")
    print(f"  class_weight={class_weight}")
    print()

    main(
        folds=folds_to_execute,
        n_components=args.components,
        C=args.reg_c,
        gamma=gamma_val,
        n_pca_components=args.pca_components,
        penalty=args.penalty,
        l1_ratio=args.l1_ratio,
        class_weight=class_weight,
        override=args.override,
    )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nElapsed time: {iara.utils.str_format_time(elapsed)}")
