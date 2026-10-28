"""
CNN MEL do artigo (Silva et al., 2025, Seção VI-A / Tabela 7: SP 63,52 ± 2,26).

Configuração do texto do artigo, igual à 1ª coluna do small_grid MEL de training_scripts/grid_search.py
(com o get_manager da CNN):
  conv_n_neurons [1024, 128], LeakyReLU, MaxPool2d(2, 2), conv_dropout 0.4, BatchNorm2d, kernel 5
  (padding 2, padrão do iara_cnn.CNN; o texto do artigo diz 1), classificador com 128 neurônios, dropout 0.4,
  sem normalização, ReLU, saída Sigmoid, Adam (lr 1e-6, weight_decay 1e-3), CrossEntropyLoss com pesos de
  classe, batch 32, 512 épocas, patience 32.
Entrada: iara_default.default_iara_mel_audio_processor() (LOG_MELGRAM, 256 bandas, 1024 pts, decimação 3,
0,512 s) em imagens de 32 janelas com 50% de sobreposição (iara_default.default_image_input()).
Folds: os 10 do 5x2cv oficial do pipeline iara.

Não confundir com test_scripts/cnn.py ('cnn'), que tem outra arquitetura (128/64, lr 1e-5, sem sobreposição).
Rodar no container iara-dev:  python test_scripts/cnn_mel_artigo.py -F 0-9
"""
import argparse
import time
import typing

import torch

import iara.utils
import iara.default as iara_default
import iara.ml.models.cnn as iara_cnn
import iara.ml.experiment as iara_exp
import iara.ml.metrics as iara_metrics
import iara.ml.models.trainer as iara_trn

from iara.default import DEFAULT_DIRECTORIES


def main(folds: typing.List[int]):

    output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/tests"
    directories = DEFAULT_DIRECTORIES

    grid = iara_metrics.GridCompiler()

    # Como no grid search do artigo: Classifier.CNN.get_input_type() -> Image(n_windows=32, overlap=0.5)
    input_type = iara_default.default_image_input()

    dp = iara_default.default_iara_mel_audio_processor(directories)

    config = iara_exp.Config(
            name = 'cnn_mel_artigo',
            dataset = iara_default.default_collection(),
            dataset_processor = dp,
            output_base_dir = output_base_dir,
            input_type = input_type)

    trainers = []
    trainers.append(iara_trn.OptimizerTrainer(
            training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
            trainer_id = 'cnn_mel_artigo',
            n_targets = config.dataset.target.get_n_targets(),
            batch_size = 32,
            n_epochs = 512,
            patience = 32,
            model_allocator = lambda input_shape, n_targets:
                    iara_cnn.CNN(
                            input_shape = input_shape,

                            conv_n_neurons = [1024, 128],
                            conv_activation = torch.nn.LeakyReLU,
                            conv_pooling = torch.nn.MaxPool2d,
                            conv_pooling_size = [2, 2],
                            conv_dropout = 0.4,
                            batch_norm = torch.nn.BatchNorm2d,
                            kernel_size = 5,
                            padding = None,

                            classification_n_neurons = [128],
                            n_targets = n_targets,
                            classification_dropout = 0.4,
                            classification_norm = None,
                            classification_hidden_activation = torch.nn.ReLU,
                            classification_output_activation = torch.nn.Sigmoid
                    ),
            optimizer_allocator=lambda model:
                    torch.optim.Adam(
                            model.parameters(),
                            weight_decay = 1e-3,
                            lr = 1e-6),
            loss_allocator = lambda class_weights:
                    torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
            ))

    manager = iara_exp.Manager(config, *trainers)

    # override=False: não move para backup um treino já existente com a mesma configuração
    result_grid = manager.run(folds = folds, override=False)

    for (eval_subset, eval_strategy), result_dict in result_grid.items():
        if eval_subset == iara_trn.Subset.ALL:
            continue

        for trainer_id, results in result_dict.items():

            for i_fold, result in enumerate(results):

                grid.add(params = {
                                'eval_strategy': eval_strategy,
                                'eval_subset': eval_subset,
                            },
                            i_fold=i_fold,
                            target=result['Target'],
                            prediction=result['Prediction'])

    print(grid)


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description='RUN CNN MEL (artigo) cross-validation')
    parser.add_argument('-F', '--fold', type=str, default=None,
                        help='Specify folds to be executed. Example: 0-9')

    args = parser.parse_args()

    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(1)))

    main(folds = folds_to_execute)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {iara.utils.str_format_time(elapsed_time)}")
