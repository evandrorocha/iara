"""
Test script for SVM with Nyström kernel approximation on the IARA dataset.

This script evaluates SVMNystroem (approximate RBF-SVM via Nyström + SGDClassifier)
as a classifier for underwater acoustic target recognition.

Method justification:
    - SVC(kernel='rbf') is infeasible for ~500K windows (O(n²) complexity)
    - Nyström approximates the RBF kernel mapping in R^n_components (m << n)
    - SGDClassifier(loss='hinge') = SVM linear in the transformed space
    - Combined: approximate RBF-SVM with O(m*n) complexity (~5 min/fold)
    - Uses InputType.Window() → trained on individual windows, evaluated by_audio

Usage (inside Docker container):
    # Run fold 0 only (quick test):
    python test_scripts/svm.py

    # Run all 10 folds (full statistical evaluation):
    python test_scripts/svm.py -F 0-9

    # Run specific folds:
    python test_scripts/svm.py -F 0,2,4

Reference:
    Williams, C., & Seeger, M. (2001). Using the Nyström method to speed up
    kernel machines. NeurIPS.
"""
import time
import typing
import argparse

import iara.utils
import iara.ml.dataset as iara_dataset
import iara.default as iara_default
import iara.ml.experiment as iara_exp
import iara.ml.metrics as iara_metrics
import iara.ml.models.trainer as iara_trn
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc

from iara.default import DEFAULT_DIRECTORIES


def main(folds: typing.List[int]):

    output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/tests"
    directories = DEFAULT_DIRECTORIES

    grid = iara_metrics.GridCompiler()

    # Use individual windows as input (same as MLP baseline)
    # The by_audio evaluation will apply majority vote across windows of each file
    input_type = iara_dataset.InputType.Window()

    # Audio preprocessing pipeline (same as MLP for fair comparison)
    dp = iara_manager.AudioFileProcessor(
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

    config = iara_exp.Config(
        name='svm_nystroem',
        dataset=iara_default.default_collection(),
        dataset_processor=dp,
        output_base_dir=output_base_dir,
        input_type=input_type
    )

    trainers = []

    # SVM with Nyström approximation (n_components=300, gamma='scale', C=1.0)
    # This is the main experiment — approximate RBF-SVM on window-level features
    trainers.append(iara_trn.SVMNystroemTrainer(
        training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
        trainer_id='svm_nystroem_300',
        n_targets=config.dataset.target.get_n_targets(),
        n_components=300,
        gamma='scale',
        C=1.0
    ))

    manager = iara_exp.Manager(config, *trainers)

    result_grid = manager.run(folds=folds, override=True)

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


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='Test SVM with Nystroem kernel approximation on IARA dataset'
    )
    parser.add_argument(
        '-F', '--fold',
        type=str,
        default=None,
        help='Folds to execute. Examples: "0" | "0-9" | "0,2,4"'
    )

    args = parser.parse_args()

    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(1)))

    print(f"Running SVM Nyström on folds: {folds_to_execute}")
    print(f"  n_components=300, gamma='scale', C=1.0")
    print(f"  InputType: Window (by_audio evaluation via majority vote)")
    print()

    main(folds=folds_to_execute)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nElapsed time: {iara.utils.str_format_time(elapsed)}")
