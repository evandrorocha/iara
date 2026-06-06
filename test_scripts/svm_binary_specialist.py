"""
Train a binary SMALL vs MEDIUM SVM specialist on hybrid MEL+LOFAR features.

The specialist is trained ONLY on SMALL (class 0) and MEDIUM (class 1) samples,
using concatenated MEL (256) + LOFAR-PCA (64) = 320-dim features. This matches
the feature space of the multiclass hybrid SVM and may reduce variance vs the
LOFAR-only specialist by providing complementary acoustic information.

Usage (inside Docker container):
    # Train all 10 folds:
    python test_scripts/svm_binary_specialist.py -F 0-9

    # Then use as pre-trained specialist in the hybrid cascade:
    python test_scripts/svm.py --cascade --analysis hybrid --components 4000 \\
        --pretrained_specialist_dir results/trainings/tests/svm_nystroem_6000_hybrid_binary_small_medium_pca64_elasticnet_l1r0.15 \\
        --pretrained_large_specialist_dir results/trainings/tests/svm_nystroem_6000_lofar_elasticnet_l1r0.15 \\
        -F 0-9
"""
import time
import typing
import argparse
import os
import sys

import pandas as pd

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svm import SVMNystroemHybridTrainer, HybridAudioFileProcessor


class SmallMediumFilter:
    """Keep only SMALL (length < 50m) and MEDIUM (50 <= length < 100m) ships.
    Drops LARGE (length >= 100m) and BACKGROUND (non-numeric length).
    Applied before the target function so the dataset has exactly n_targets=2 classes.
    """
    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        lengths = pd.to_numeric(input_df['Length'], errors='coerce')
        return input_df[lengths < 100]


def main(
    folds: typing.List[int],
    n_components: int = 6000,
    C: float = 1.0,
    gamma: typing.Union[str, float] = 'scale',
    n_pca_components: int = 64,
    penalty: str = 'elasticnet',
    l1_ratio: float = 0.15,
    class_weight: typing.Union[str, None] = 'balanced',
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

    # Collection with only SMALL (0) and MEDIUM (1).
    # SmallMediumFilter (length < 100m) runs first to drop LARGE and BACKGROUND rows,
    # so GenericTarget(n_targets=2) sees only 2 classes and get_n_targets() returns 2.
    binary_collection = iara.records.CustomCollection(
        collection=iara.records.Collection.OS,
        target=iara.records.GenericTarget(
            n_targets=2,
            function=iara_default.Target.classify_row,
            include_others=False
        ),
        filters=SmallMediumFilter(),
        only_sample=False
    )

    name_parts = [f'svm_nystroem_{n_components}', 'hybrid', 'binary_small_medium',
                  f'pca{n_pca_components}']
    if penalty != 'l2':
        name_parts.append(penalty)
        if penalty == 'elasticnet':
            name_parts.append(f'l1r{l1_ratio}')
    if C != 1.0:
        name_parts.append(f'C{C}')
    exp_name = '_'.join(name_parts)

    config = iara_exp.Config(
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
    result_grid = manager.run(folds=folds, override=False)

    for (eval_subset, eval_strategy), result_dict in result_grid.items():
        if eval_subset == iara_trn.Subset.ALL:
            continue
        for trainer_id, results in result_dict.items():
            for i_fold, result in enumerate(results):
                grid.add(
                    params={
                        'eval_strategy': eval_strategy,
                        'eval_subset': eval_subset,
                    },
                    i_fold=i_fold,
                    target=result['Target'],
                    prediction=result['Prediction']
                )

    print(grid)

    spec_dir = os.path.join(output_base_dir, exp_name)
    large_spec_dir = os.path.join(output_base_dir,
                                  'svm_nystroem_6000_lofar_elasticnet_l1r0.15')
    print(f"\n[Done] Binary hybrid specialist saved to: {spec_dir}")
    print(f"\n[Cascade] To use this specialist in the hybrid cascade:")
    print(f"  python test_scripts/svm.py --cascade --analysis hybrid --components 4000 \\")
    print(f"    --pretrained_specialist_dir {spec_dir} \\")
    print(f"    --pretrained_large_specialist_dir {large_spec_dir} \\")
    print(f"    -F 0-9")


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='Train binary SMALL vs MEDIUM SVM specialist on hybrid MEL+LOFAR features.'
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
        '--pca_components', type=int, default=64,
        help='Number of PCA components applied to LOFAR portion. Default: 64'
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

    args = parser.parse_args()

    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(1)))

    gamma_val = args.gamma
    try:
        gamma_val = float(args.gamma)
    except ValueError:
        pass

    class_weight = None if args.no_balanced else 'balanced'

    print(f"[Binary Specialist] SMALL vs MEDIUM on HYBRID (MEL+LOFAR) features")
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
    )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nElapsed time: {iara.utils.str_format_time(elapsed)}")
