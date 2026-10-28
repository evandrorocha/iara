# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
**Abordagem Proposta: SVM Nyström Gaussiano com Penalização ElasticNet e Opção de Rejeição**

> **Atualização de 20/10/2026.** Todos os números deste relatório são gerados por
> `reports/atualizar_numeros_comparacao_modelos.py` a partir das predições por janela em disco
> (`results/trainings/{tests,multirep,cpa_exp}/`), com a mesma convenção do pipeline iara e de `notebooks/iara_results.py`:
> voto majoritário por arquivo (`Counter.most_common`), **ACC = acurácia balanceada** (média dos recalls), F1 macro,
> média ± desvio-padrão populacional (ddof=0) nos 10 folds e **F-test 5×2cv de Alpaydin** (o teste do artigo) para toda
> afirmação de "melhor/supera". Valores do artigo vêm da Tabela 7 (*Silva et al., 2025*). Versões anteriores deste
> relatório tinham números de cascatas com **vazamento** (especialista SMALL×LARGE; ver Seção 9), mistura de acurácia
> micro e balanceada, um "SP" calculado só como média geométrica (Tabela 2) e valores 3C sem experimento correspondente
> (Tabela 5); isso foi corrigido abaixo.

---

## 1. Introdução e Metodologia Experimental

Este documento apresenta uma análise comparativa entre o modelo proposto de **Support Vector Machines com Aproximação de Nyström (RBF-SVM)** e os modelos baselines do artigo de referência do dataset IARA (*Silva et al., 2025*): *Random Forest (RF)*, *Multi-Layer Perceptron (MLP)* e *Convolutional Neural Network (CNN)*.

### Metodologia de Validação Cruzada (5x2cv):
Adotou-se o protocolo de **Validação Cruzada 5x2 (5x2cv)**, totalizando 10 folds, sobre os 1706 áudios da coleção usada. A partição é estratificada e restrita no nível de **gravação de áudio (`ID`)** (`exclusive_header='ID'` em `src/iara/ml/experiment.py`), como no artigo: janelas de um mesmo arquivo nunca ficam simultaneamente em treino e teste. O isolamento **não** é por navio: gravações diferentes do mesmo navio podem cair em treino e teste (a auditoria `reports/auditoria_claude_iara.md` mediu que 63,5% das gravações de navio no teste têm o mesmo navio no treino). A antiga menção a "*Exclusive Ships on Test*" referia-se ao nome do parâmetro do código, não a um isolamento por navio.

---

## 2. Desempenho de Classificação Global (Cobertura de 100%)

A Tabela 1 consolida as métricas globais na partição de teste para as classes `SMALL`, `MEDIUM`, `LARGE` e `BACKGROUND`.

### Tabela 1: Métricas de Desempenho Geral no Conjunto de Teste (Sem Opção de Rejeição)

| Classificador / Arquitetura | Representação | Índice SP (%) | Acurácia Balanceada (ACC) (%) | F1-Score (Macro) (%) | Fonte |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **RF** | MEL | 62.22 ± 1.86 | 62.64 ± 1.84 | 63.79 ± 1.73 | Tabela 7 do artigo (Silva et al., 2025) |
| **RF** | LOFAR | 56.92 ± 1.69 | 58.87 ± 1.68 | 58.00 ± 1.41 | Tabela 7 do artigo (Silva et al., 2025) |
| **MLP** | MEL | 63.38 ± 1.81 | 64.51 ± 1.75 | 62.89 ± 1.68 | Tabela 7 do artigo (Silva et al., 2025) |
| **MLP** | LOFAR | 66.51 ± 1.39 | 67.48 ± 1.24 | 66.72 ± 1.17 | Tabela 7 do artigo (Silva et al., 2025) |
| **CNN** | MEL | 63.52 ± 2.26 | 64.99 ± 2.09 | 63.04 ± 2.02 | Tabela 7 do artigo (Silva et al., 2025) |
| **CNN** | LOFAR | 66.05 ± 1.90 | 67.02 ± 1.78 | 66.29 ± 2.13 | Tabela 7 do artigo (Silva et al., 2025) |
| **MLP (re-execução local)** | MEL | 64.74 ± 2.05 | 65.73 ± 1.90 | 64.19 ± 2.14 | `mlp` |
| **MLP (re-execução local)** | LOFAR | 66.24 ± 1.78 | 67.22 ± 1.65 | 66.42 ± 1.87 | `mlp_lofar` |
| **CNN (re-execução local)** | MEL | 63.80 ± 2.23 | 65.01 ± 2.05 | 63.29 ± 2.25 | `cnn` |
| **MLP (pca64)** | HÍBRIDO | 66.13 ± 1.09 | 66.92 ± 1.06 | 65.92 ± 1.13 | `mlp_hybrid_pca64` |
| *---* | *---* | *---* | *---* | *---* | *---* |
| **SVM (m=300)** | MEL | 59.96 ± 3.38 | 62.34 ± 1.98 | 59.47 ± 3.16 | `svm_nystroem` |
| **SVM (m=1000)** | MEL | 62.96 ± 2.08 | 64.11 ± 1.98 | 62.74 ± 1.87 | `svm_nystroem_1000` |
| **SVM (m=1000) (norm + pca64)** | MEL | 60.83 ± 1.67 | 61.86 ± 1.66 | 60.95 ± 1.31 | `svm_nystroem_1000_log_melgram_norm_pca64` |
| **SVM (m=2000)** | MEL | 63.04 ± 2.37 | 64.45 ± 1.93 | 62.93 ± 2.22 | `svm_nystroem_2000_log_melgram` |
| **SVM (m=2000) (C=10)** | MEL | 62.18 ± 3.14 | 63.53 ± 2.50 | 62.67 ± 2.84 | `svm_nystroem_2000_log_melgram_C10.0` |
| **SVM (m=3000)** | MEL | 63.65 ± 1.99 | 64.77 ± 1.87 | 63.59 ± 1.92 | `svm_nystroem_3000_log_melgram` |
| **SVM (m=4000) (C=2.0) (pca64) (elasticnet)** | MEL | 63.82 ± 1.61 | 64.61 ± 1.60 | 64.60 ± 1.54 | `svm_nystroem_4000_log_melgram_pca64_elasticnet_l1r0.15_C2.0` |
| **SVM (m=4000) (C=2.0) (elasticnet) — "Golden MEL"** | MEL | 63.78 ± 1.14 | 64.56 ± 1.18 | 64.32 ± 1.02 | `svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0` |
| **SVM (m=4000) (C=0.5) (elasticnet)** | MEL | 63.48 ± 1.50 | 64.90 ± 1.23 | 62.97 ± 1.75 | `svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C0.5` |
| *---* | *---* | *---* | *---* | *---* | *---* |
| **SVM (m=1000)** | LOFAR | 57.86 ± 3.83 | 60.95 ± 2.89 | 57.86 ± 3.53 | `svm_nystroem_1000_lofar` |
| **SVM (m=2000)** | LOFAR | 60.10 ± 2.25 | 62.54 ± 1.98 | 59.95 ± 2.77 | `svm_nystroem_2000_lofar` |
| **SVM (m=2000) (C=0.1)** | LOFAR | 54.84 ± 1.60 | 59.67 ± 1.28 | 54.64 ± 1.69 | `svm_nystroem_2000_lofar_C0.1` |
| **SVM (m=2000) (C=2.0)** | LOFAR | 60.15 ± 2.74 | 62.45 ± 2.17 | 60.59 ± 2.64 | `svm_nystroem_2000_lofar_C2.0` |
| **SVM (m=4000) (C=2.0)** | LOFAR | 61.70 ± 3.67 | 63.55 ± 2.91 | 62.85 ± 3.33 | `svm_nystroem_4000_lofar_C2.0` |
| **SVM (m=1000) (C=2.0) (pca64)** | LOFAR | 61.96 ± 2.22 | 63.16 ± 2.18 | 62.65 ± 1.74 | `svm_nystroem_1000_lofar_pca64_C2.0` |
| **SVM (m=4000) (C=2.0) (pca64) (elasticnet)** | LOFAR | 63.10 ± 2.19 | 64.04 ± 1.88 | 64.34 ± 2.23 | `svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0` |
| **SVM (m=4000) (C=0.5) (pca64) (elasticnet)** | LOFAR | 63.81 ± 2.35 | 64.89 ± 2.05 | 64.60 ± 2.38 | `svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C0.5` |
| **SVM (m=4000) (C=0.5) (elasticnet)** | LOFAR | 60.66 ± 2.78 | 63.23 ± 2.05 | 60.82 ± 2.47 | `svm_nystroem_4000_lofar_elasticnet_l1r0.15_C0.5` |
| **SVM (m=6000) (elasticnet)** | LOFAR | 63.83 ± 1.91 | 65.47 ± 1.77 | 63.58 ± 1.72 | `svm_nystroem_6000_lofar_elasticnet_l1r0.15` |
| *---* | *---* | *---* | *---* | *---* | *---* |
| **SVM (m=1000), max_iter=1000** | DEMON | 39.15 ± 14.70 | 49.93 ± 2.88 | 45.51 ± 5.01 | `svm_nystroem_1000_demon_C2.0_elasticnet_l1r0.15_it1000_4c` |
| **SVM (m=4000), max_iter=1000** | DEMON | 45.52 ± 5.75 | 53.11 ± 2.18 | 47.30 ± 4.22 | `svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_it1000_4c` |
| **SVM (m=6000), max_iter=1000** | DEMON | 44.95 ± 4.58 | 52.73 ± 1.86 | 46.99 ± 2.63 | `svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000_4c` |
| **RF (100 árvores)** | DEMON | 44.44 ± 2.24 | 46.08 ± 1.79 | 47.16 ± 1.94 | `rf_demon_n100_balanced_4c` |
| **MLP [32,16] (25 ép.)** | DEMON | 51.76 ± 2.75 | 54.04 ± 2.54 | 50.39 ± 3.25 | `mlp_demon_h32-16_e25_4c` |
| **CNN 2D ad hoc [COLAPSADA]** | DEMON | 36.28 ± 6.38 | 44.48 ± 2.03 | 39.46 ± 3.32 | `cnn2d_demon_w32_b32_e20_4c` |
| *---* | *---* | *---* | *---* | *---* | *---* |
| **RF (100 árvores)** | HHT | 57.64 ± 2.14 | 58.11 ± 2.03 | 60.21 ± 2.04 | `rf_hht_n100_balanced_4c` |
| **MLP [32,16]** | HHT | 56.31 ± 1.77 | 58.09 ± 1.34 | 55.04 ± 1.54 | `mlp_hht_h32-16_e30_noseed_4c` |
| **SVM (m=6000) (elasticnet)** | HHT | 51.18 ± 6.09 | 54.58 ± 2.36 | 52.80 ± 5.17 | `svm_nystroem_6000_hht_C2.0_elasticnet_l1r0.15_4c` |
| **CNN 2D ad hoc [COLAPSADA]** | HHT | 38.25 ± 7.88 | 44.29 ± 4.01 | 38.31 ± 6.63 | `cnn2d_hht_w32_b32_e25_4c` |
| *---* | *---* | *---* | *---* | *---* | *---* |
| **MLP [64,32] Fusão (MEL 5,12 s + DEMON)** | MULTIMODAL | 66.92 ± 1.59 | 67.40 ± 1.58 | 66.99 ± 1.54 | `mlp_mel5s-demon_h64-32_e35_4c` |
| **SVM (m=6000) Fusão (MEL 5,12 s + DEMON)** | MULTIMODAL | 64.67 ± 1.41 | 65.60 ± 1.27 | 64.91 ± 1.51 | `svm_nystroem_6000_mel5s-demon_C2.0_elasticnet_l1r0.15_4c` |
| **RF Fusão (MEL 5,12 s + DEMON)** | MULTIMODAL | 62.04 ± 1.66 | 62.15 ± 1.67 | 63.72 ± 1.46 | `rf_mel5s-demon_n100_balanced_4c` |
| **CNN 2D ad hoc Fusão [COLAPSADA]** | MULTIMODAL | 8.60 ± 14.13 | 39.57 ± 10.26 | 32.12 ± 11.45 | `cnn2d_mel5s-demon_w32_b32_e25_4c` |
| **SVM (m=4000) (C=2.0) (elasticnet, norm)** | HÍBRIDO | 64.67 ± 2.24 | 65.31 ± 1.90 | 65.83 ± 2.43 | `svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C2.0` |
| **SVM (m=4000) (C=0.5) (elasticnet, norm)** | HÍBRIDO | 65.16 ± 1.67 | 65.77 ± 1.55 | 65.75 ± 1.51 | `svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5` |
| *---* | *---* | *---* | *---* | *---* | *---* |
| **SVM Cascata: estágio 1 → SM(PCA64)/LOFAR (só MEDIUM)** | HÍBRIDO | 65.13 ± 1.98 | 65.51 ± 1.98 | 65.39 ± 2.06 | `svm_nystroem_4000_hybrid_cascade_pretrained_svm_nystroem_6000_hybrid_binary_small_medium_pca64_elasticnet_l1r0.15_large_svm_nystroem_6000_lofar_elasticnet_l1r0.15` |
| **SVM Cascata: estágio 1 → SM(PCA64)/LOFAR (SMALL+MEDIUM)** | HÍBRIDO | 65.33 ± 1.91 | 65.73 ± 1.87 | 65.29 ± 2.04 | `svm_nystroem_4000_hybrid_cascade_pretrained_svm_nystroem_6000_hybrid_binary_small_medium_pca64_elasticnet_l1r0.15_smallalso_large_svm_nystroem_6000_lofar_elasticnet_l1r0.15` |
| **SVM Cascata: MEL → SM(PCA64)/LOFAR (SMALL+MEDIUM)** | HÍBRIDO | 63.47 ± 1.39 | 64.39 ± 1.36 | 63.67 ± 1.46 | `svm_nystroem_4000_log_melgram_cascade_pretrained_svm_nystroem_6000_hybrid_binary_small_medium_pca64_elasticnet_l1r0.15_smallalso_large_svm_nystroem_6000_lofar_elasticnet_l1r0.15` |
| **SVM Cascata: HYBRID-MC → SM(PCA64)+SL(PCA8) (sempre), SL retreinado** | HÍBRIDO | 65.77 ± 1.23 | 65.96 ± 1.23 | 66.79 ± 1.21 | `svm_nystroem_4000_hybrid_cascade_pretrained_sm6000pca64_smallalso_fbmc_large_sl6000pca8_hybmc4000C2.0_elasticnet_l1r0.15` |
| **SVM Cascata: HYBRID-MC → SM(PCA64)+SL(PCA8) (runner-up), SL retreinado** | HÍBRIDO | 65.87 ± 1.41 | 66.08 ± 1.39 | 66.89 ± 1.41 | `svm_nystroem_4000_hybrid_cascade_pretrained_sm6000pca64_smallalso_fbmc_large_sl6000pca8_hybmc4000C2.0_slro_elasticnet_l1r0.15` |
| **SVM Cascata: LOFAR-MC → SL(PCA8), SL retreinado** | HÍBRIDO | 67.16 ± 1.82 | 67.63 ± 1.87 | 66.82 ± 1.35 | `svm_nystroem_4000_hybrid_cascade_pretrained_sm6000pca64_smallalso_fbmc_large_sl6000pca8_lofarmc6000_nosm_elasticnet_l1r0.15` |
| **SVM Cascata: LOFAR-MC → SM só MEDIUM + SL(PCA8), SL retreinado** | HÍBRIDO | 64.93 ± 1.81 | 65.92 ± 1.76 | 65.37 ± 1.40 | `svm_nystroem_4000_hybrid_cascade_pretrained_sm6000pca64_smallalso_fbmc_large_sl6000pca8_lofarmc6000_smonly_elasticnet_l1r0.15` |
| **Cascata por arquivo: LOFAR-MC → SL(PCA8), SL retreinado** | HÍBRIDO | 66.00 ± 1.90 | 66.77 ± 1.92 | 65.45 ± 1.76 | `cascade_file_lofarmc6000_sl6000pca8_4c` |
| **SVM Cascata Dupla com especialista LOFAR (t=0.55)** | HÍBRIDO | 64.33 ± 2.12 | 64.90 ± 1.94 | 64.70 ± 2.28 | `svm_nystroem_4000_hybrid_cascade_pretrained_svm_nystroem_6000_lofar_elasticnet_l1r0.15_elasticnet_l1r0.15_ct0.55` |
| **SVM Cascata: MEL→SM(PCA64)+SL(PCA8) (SMALL+MEDIUM)** | HÍBRIDO | **INVALIDADA** | — | — | especialista SL com vazamento; ver Seção 9 |

> [!NOTE]
> *As incertezas são o desvio-padrão populacional entre os 10 folds. As linhas "Tabela 7 do artigo" são os valores publicados (sem resultados por fold em disco); todas as demais são recalculadas das predições em disco. O MLP LOFAR re-executado localmente (`mlp_lofar`: SP 66.24 ± 1.78, ACC 67.22 ± 1.65) difere do artigo em 0.27 p.p. de SP e 0.26 p.p. de ACC. As linhas DEMON, HHT e de fusão vêm de scripts ad hoc (`results/trainings/multirep`), com vetores a cada 5,12 s (o MEL dessas linhas é o cache `mel5s`, com resolução temporal 10× menor que a das features oficiais de 0,512 s). As linhas marcadas **[COLAPSADA]** são CNNs ad hoc com folds em que uma classe tem recall ≈ 0 (SP baixo e desvio alto); não servem como baseline. Os SVM DEMON usam `max_iter=1000` sem early stopping. A coluna "F1" da versão anterior misturava F1 micro e acurácia; agora é F1 macro em todas as linhas.*

---

## 3. Análise da Variância Estatística e Sensibilidade Multiclasse

1. **Equivalência com a CNN, com menor desvio:** O **SVM (m=4000, "Golden MEL")** alcançou **63.78 ± 1.14%** de SP, próximo da CNN MEL do artigo (**63.52 ± 2.26%**). Contra a re-execução local da CNN MEL nos mesmos folds (`cnn`: SP 63.80 ± 2.23), o F-test 5×2cv dá p = 0.265 em SP e p = 0.292 em ACC: **sem diferença significativa**. O desvio entre folds do SVM é ≈ 0.51× o da CNN do artigo (≈ 0.26× a variância); a versão anterior dizia "metade da variância", mas a razão de ~0,5 é a dos **desvios**.
2. **PCA sem efeito significativo (MEL e LOFAR):** No MEL m=4000, o PCA64 dá SP 63.82 ± 1.61 contra 63.78 ± 1.14 sem PCA (p = 0.395); no LOFAR m=4000, PCA64+ElasticNet dá SP 63.10 ± 2.19 contra 61.70 ± 3.67 sem PCA (p = 0.143 em SP; p = 0.197 em ACC). O SVM LOFAR m=6000 **sem** PCA tem ACC 65.47 ± 1.77 contra 64.04 ± 1.88 do m=4000 com PCA64 (p = 0.533). Portanto, as afirmações anteriores de que o PCA "degrada" o MEL ("dupla compressão") e é "essencial" no LOFAR ("filtro passa-faixa estatístico") **não são sustentadas pelos dados**; a escolha "MEL sem PCA / LOFAR com PCA64" é uma configuração adotada, não uma conclusão estatística.

---

## 4. Opção de Rejeição (Filtro de Confiança)

Propõe-se um **operador de decisão com opção de rejeição** baseado na concordância temporal das janelas. Seja um arquivo fatiado em $W$ janelas e $n_k$ o número de janelas votadas na classe $k$. A confiança do arquivo é $c(x) = \max_k n_k / W$ (fração de janelas que votaram na classe vencedora); o arquivo é aceito se $c(x) \ge t$. Nenhuma probabilidade/softmax é usada. Sobre os arquivos aceitos de cada fold calculam-se cobertura, ACC balanceada, acurácia micro e SP (fórmula do artigo); reporta-se média ± desvio (ddof=0).

> Diferenças em relação à versão anterior desta tabela (`scratch/scratch_rejection_10folds.py`): lá a coluna "ACC" era a acurácia **micro**, o "SP" era só a média geométrica dos recalls (não o SP do artigo) e o desvio usava ddof=1. As linhas "+ bias" (antes "Calibrado") somam um vetor fixo de bias aos escores de SMALL/MEDIUM (`scratch/scratch_generate_calibrated_csvs*.py`); **não** é calibração de probabilidade, e não está documentado neste relatório em que conjunto (validação ou teste) esses vetores foram escolhidos.

### Tabela 2: Curva de Trade-off Cobertura-Acurácia

| Limiar ($t$) | Modelo | Cobertura (%) | ACC balanceada (%) | Acurácia micro (%) | Índice SP (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| $t \ge 0.0$ *(sem rejeição)* | SVM MEL (m=4000, C=2.0) | 100.0 ± 0.0 | 64.6 ± 1.2 | 63.3 ± 1.0 | 63.8 ± 1.1 |
| $t \ge 0.0$ *(sem rejeição)* | SVM LOFAR (m=4000, C=0.5, PCA64) | 100.0 ± 0.0 | 64.9 ± 2.0 | 63.8 ± 2.1 | 63.8 ± 2.4 |
| $t \ge 0.0$ *(sem rejeição)* | SVM LOFAR (m=4000, C=2.0, PCA64) | 100.0 ± 0.0 | 64.0 ± 1.9 | 63.3 ± 2.1 | 63.1 ± 2.2 |
| $t \ge 0.0$ *(sem rejeição)* | MLP MEL (local) | 100.0 ± 0.0 | 65.7 ± 1.9 | 63.5 ± 1.9 | 64.7 ± 2.0 |
| $t \ge 0.0$ *(sem rejeição)* | MLP LOFAR (local) | 100.0 ± 0.0 | 67.2 ± 1.6 | 65.5 ± 1.7 | 66.2 ± 1.8 |
| $t \ge 0.0$ *(sem rejeição)* | MLP Híbrido (pca64) | 100.0 ± 0.0 | 66.9 ± 1.1 | 64.9 ± 1.2 | 66.1 ± 1.1 |
| $t \ge 0.0$ *(sem rejeição)* | SVM Híbrido (C=0.5) | 100.0 ± 0.0 | 65.8 ± 1.5 | 64.8 ± 1.3 | 65.2 ± 1.7 |
| $t \ge 0.0$ *(sem rejeição)* | SVM Híbrido (C=0.5) + bias [0.40, −0.19, 0, 0] | 100.0 ± 0.0 | 65.0 ± 1.7 | 64.6 ± 1.7 | 64.5 ± 2.0 |
| $t \ge 0.0$ *(sem rejeição)* | SVM LOFAR (C=0.5, PCA64) + bias [0.32, 0.27, 0, 0] | 100.0 ± 0.0 | 64.4 ± 2.3 | 63.2 ± 2.1 | 63.9 ± 2.4 |
| $t \ge 0.0$ *(sem rejeição)* | CNN MEL (local; 7 segmentos/arquivo) | 100.0 ± 0.0 | 65.0 ± 2.0 | 62.0 ± 2.3 | 63.8 ± 2.2 |
| $t \ge 0.5$ *(maioria simples)* | SVM MEL (m=4000, C=2.0) | 89.7 ± 1.2 | 66.9 ± 1.2 | 66.2 ± 1.5 | 66.1 ± 1.6 |
| $t \ge 0.5$ *(maioria simples)* | SVM LOFAR (m=4000, C=0.5, PCA64) | 78.3 ± 1.6 | 69.9 ± 2.6 | 70.6 ± 2.6 | 68.8 ± 3.1 |
| $t \ge 0.5$ *(maioria simples)* | SVM LOFAR (m=4000, C=2.0, PCA64) | 78.2 ± 1.9 | 69.6 ± 2.3 | 70.4 ± 2.1 | 68.6 ± 2.9 |
| $t \ge 0.5$ *(maioria simples)* | MLP MEL (local) | 85.6 ± 1.2 | 68.4 ± 1.6 | 67.4 ± 1.7 | 67.3 ± 1.7 |
| $t \ge 0.5$ *(maioria simples)* | MLP LOFAR (local) | 77.4 ± 1.8 | 72.5 ± 1.6 | 72.6 ± 1.4 | 71.4 ± 1.9 |
| $t \ge 0.5$ *(maioria simples)* | MLP Híbrido (pca64) | 86.2 ± 1.4 | 70.0 ± 1.6 | 69.1 ± 1.8 | 69.1 ± 1.6 |
| $t \ge 0.5$ *(maioria simples)* | SVM Híbrido (C=0.5) | 88.5 ± 1.6 | 68.3 ± 2.2 | 67.9 ± 1.9 | 67.6 ± 2.3 |
| $t \ge 0.5$ *(maioria simples)* | SVM Híbrido (C=0.5) + bias [0.40, −0.19, 0, 0] | 91.0 ± 1.7 | 67.0 ± 1.9 | 66.9 ± 1.8 | 66.5 ± 2.2 |
| $t \ge 0.5$ *(maioria simples)* | SVM LOFAR (C=0.5, PCA64) + bias [0.32, 0.27, 0, 0] | 79.5 ± 2.7 | 69.1 ± 3.0 | 68.9 ± 3.0 | 68.5 ± 3.2 |
| $t \ge 0.5$ *(maioria simples)* | CNN MEL (local; 7 segmentos/arquivo) | 95.9 ± 1.0 | 65.8 ± 2.1 | 63.2 ± 2.4 | 64.6 ± 2.3 |
| $t \ge 0.6$ *(maioria absoluta)* | SVM MEL (m=4000, C=2.0) | 76.1 ± 2.7 | 69.2 ± 1.7 | 69.6 ± 1.9 | 68.3 ± 2.4 |
| $t \ge 0.6$ *(maioria absoluta)* | SVM LOFAR (m=4000, C=0.5, PCA64) | 62.9 ± 2.0 | 73.8 ± 3.4 | 75.9 ± 3.2 | 72.6 ± 3.9 |
| $t \ge 0.6$ *(maioria absoluta)* | SVM LOFAR (m=4000, C=2.0, PCA64) | 61.2 ± 2.2 | 74.0 ± 3.9 | 76.5 ± 3.3 | 72.9 ± 4.9 |
| $t \ge 0.6$ *(maioria absoluta)* | MLP MEL (local) | 70.4 ± 2.5 | 71.3 ± 1.8 | 72.0 ± 2.0 | 70.0 ± 2.1 |
| $t \ge 0.6$ *(maioria absoluta)* | MLP LOFAR (local) | 61.0 ± 1.6 | 76.0 ± 2.2 | 78.1 ± 2.2 | 74.7 ± 2.6 |
| $t \ge 0.6$ *(maioria absoluta)* | MLP Híbrido (pca64) | 71.3 ± 1.1 | 73.2 ± 1.6 | 73.7 ± 1.7 | 72.2 ± 1.8 |
| $t \ge 0.6$ *(maioria absoluta)* | SVM Híbrido (C=0.5) | 75.7 ± 1.5 | 71.0 ± 2.2 | 71.5 ± 1.8 | 70.2 ± 2.4 |
| $t \ge 0.6$ *(maioria absoluta)* | SVM Híbrido (C=0.5) + bias [0.40, −0.19, 0, 0] | 77.6 ± 2.4 | 69.9 ± 2.5 | 70.2 ± 2.3 | 69.3 ± 2.8 |
| $t \ge 0.6$ *(maioria absoluta)* | SVM LOFAR (C=0.5, PCA64) + bias [0.32, 0.27, 0, 0] | 62.3 ± 2.0 | 74.5 ± 3.2 | 75.2 ± 3.0 | 73.9 ± 3.6 |
| $t \ge 0.6$ *(maioria absoluta)* | CNN MEL (local; 7 segmentos/arquivo) | 82.9 ± 1.7 | 68.2 ± 2.0 | 66.6 ± 2.3 | 66.8 ± 2.3 |
| $t \ge 0.9$ *(consenso crítico)* | SVM MEL (m=4000, C=2.0) | 37.5 ± 2.3 | 77.1 ± 2.5 | 83.4 ± 2.1 | 75.6 ± 3.2 |
| $t \ge 0.9$ *(consenso crítico)* | SVM LOFAR (m=4000, C=0.5, PCA64) | 25.1 ± 1.3 | 81.5 ± 4.3 | 89.6 ± 2.8 | 79.7 ± 6.1 |
| $t \ge 0.9$ *(consenso crítico)* | SVM LOFAR (m=4000, C=2.0, PCA64) | 22.3 ± 2.9 | 80.9 ± 5.3 | 90.4 ± 2.9 | 78.8 ± 7.2 |
| $t \ge 0.9$ *(consenso crítico)* | MLP MEL (local) | 32.9 ± 2.2 | 75.8 ± 4.1 | 85.9 ± 2.8 | 72.9 ± 5.1 |
| $t \ge 0.9$ *(consenso crítico)* | MLP LOFAR (local) | 24.8 ± 2.4 | 81.2 ± 3.8 | 91.6 ± 2.1 | 78.7 ± 5.2 |
| $t \ge 0.9$ *(consenso crítico)* | MLP Híbrido (pca64) | 33.8 ± 1.4 | 80.1 ± 2.9 | 87.7 ± 1.8 | 78.6 ± 3.7 |
| $t \ge 0.9$ *(consenso crítico)* | SVM Híbrido (C=0.5) | 37.7 ± 1.3 | 76.6 ± 3.2 | 83.2 ± 2.2 | 74.8 ± 4.1 |
| $t \ge 0.9$ *(consenso crítico)* | SVM Híbrido (C=0.5) + bias [0.40, −0.19, 0, 0] | 37.8 ± 1.4 | 77.2 ± 3.5 | 81.5 ± 3.0 | 76.5 ± 3.9 |
| $t \ge 0.9$ *(consenso crítico)* | SVM LOFAR (C=0.5, PCA64) + bias [0.32, 0.27, 0, 0] | 22.7 ± 1.6 | 83.9 ± 5.3 | 89.1 ± 4.2 | 82.9 ± 6.5 |
| $t \ge 0.9$ *(consenso crítico)* | CNN MEL (local; 7 segmentos/arquivo) | 54.0 ± 2.5 | 72.0 ± 2.6 | 74.4 ± 2.1 | 70.2 ± 3.2 |

---

## 5. Análise dos Resultados de Rejeição e Discussão

As interpretações físicas abaixo (propagação, SNR, harmônicas) são **hipóteses** compatíveis com os números, não foram testadas diretamente.

### 5.1 LOFAR (SVM m=4000, C=0.5, PCA64)
* Sob $t \ge 0.9$, a ACC balanceada nos arquivos aceitos é **81.55 ± 4.30%** (acurácia micro 89.62 ± 2.79%), com cobertura de **25.10 ± 1.34%**. Com C=2.0 (`svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0`) os valores são ACC balanceada 80.90 ± 5.31%, acurácia micro 90.43 ± 2.95% e cobertura 22.27 ± 2.86%. Os valores citados antes ("90,43%" e "22,27%") eram a acurácia **micro** e a cobertura deste modelo C=2.0, não a ACC balanceada.
* Hipótese: as harmônicas discretas do LOFAR se degradam com a distância/SNR, o que reduziria a concordância temporal e, portanto, a cobertura.

### 5.2 MEL (SVM m=4000, C=2.0)
* Sob maioria absoluta ($t \ge 0.6$), o SVM MEL retém **76.07 ± 2.69%** dos arquivos com ACC balanceada de 69.25 ± 1.68%. Em $t \ge 0.9$, a ACC balanceada é 77.10 ± 2.55% (cobertura 37.48 ± 2.32%), abaixo do LOFAR no mesmo limiar, mas com cobertura maior.
* Hipótese: a integração em bandas Mel suaviza flutuações (mais cobertura) e funde harmônicas próximas (menos acurácia no topo de certeza).

### 5.3 CNN: índice de concordância mais grosseiro (não "superconfiança")
* Sob $t \ge 0.9$, a CNN MEL local mantém cobertura de **54.01 ± 2.47%** com ACC balanceada de **72.00 ± 2.63%**. A comparação no mesmo $t$ não é justa: a concordância da CNN é calculada sobre 7 segmentos por arquivo, enquanto os modelos por janela usam ≈ 235 janelas (mediana). Entre os limiares da Tabela 2, o do SVM MEL com cobertura mais próxima da CNN em $t \ge 0.9$ é $t \ge 0.9$ (cobertura 37.48 ± 2.32%, ACC 77.10 ± 2.55%); as coberturas continuam diferentes, e a afirmação antiga de "desvantagem de 16% / 9%" da CNN comparava acurácias micro em coberturas distintas.
* Como nenhuma probabilidade é usada, estes dados **não** permitem conclusões sobre calibração ou "superconfiança" da Softmax; as afirmações anteriores nesse sentido (incl. "estimador de incerteza infinitamente mais calibrado") foram removidas.

### 5.4 MLP MEL (re-execução local)
* Sob $t \ge 0.9$, a ACC balanceada do MLP MEL é **75.75 ± 4.06%** (cobertura 32.90 ± 2.17%; SVM MEL: 77.10 ± 2.55% com cobertura 37.48 ± 2.32%). Os modelos aceitam conjuntos de arquivos diferentes, então a diferença de acurácia residual não é comparável diretamente nem foi testada. A afirmação anterior de "calibração superior" do MLP não é sustentada (nenhuma probabilidade é usada).

### 5.5 Fusão espectral híbrida no MLP (Mel + LOFAR-PCA64)
* **Desempenho geral:** O MLP Híbrido (`mlp_hybrid_pca64`) atingiu **66.13 ± 1.09%** de SP e **66.92 ± 1.06%** de ACC. Contra o SVM Híbrido C=0.5 (SP 65.16 ± 1.67): p = 0.186, **sem diferença significativa**; contra o MLP LOFAR local: p = 0.668, **sem diferença significativa**.
* **Rejeição:** Sob $t \ge 0.9$, o MLP Híbrido tem ACC balanceada de 80.13 ± 2.95%, cobertura de 33.85 ± 1.44% e SP de 78.59 ± 3.67%. Em relação ao MLP MEL no mesmo limiar, a média é maior em ACC, maior em acurácia micro e maior em SP (sem teste de significância).

### 5.6 O Extrator DEMON (SVM, RF, MLP e CNN)
O DEMON (253 bins de 0,39 a 25 Hz, um vetor a cada 5,12 s) foi avaliado nos 10 folds oficiais em quatro famílias:
* **MLP [32,16] (25 ép.):** SP 51.76 ± 2.75%, ACC 54.04 ± 2.54%. Contra o SVM m=6000 (SP 44.95 ± 4.58): p = 0.064, **sem diferença significativa**; contra o RF (SP 44.44 ± 2.24): p = 0.047, diferença **significativa**. Recall de BACKGROUND 78.78%; fração de SMALL predita como LARGE: 42.4% no SVM m=6000 e 22.5% no MLP. Com o modelo retreinado em 3 classes (`mlp_demon_h32-16_e25_3c`), SP 55.50 ± 2.61% (SMALL 63.0%, MEDIUM 46.5%, LARGE 58.6%).
* **Random Forest:** SP 44.44 ± 2.24%; recall de SMALL 52.47% (4 classes) e 60.06% (3 classes, retreinado); 48.6% dos arquivos de BACKGROUND são preditos SMALL e o recall de BACKGROUND é 27.96%.
* **CNN 2D ad hoc:** SP 36.28 ± 6.38% — **colapsada** (desvio alto, folds com classe zerada). O resultado indica falha de treino dessa CNN ad hoc; não permite concluir que "convoluções espaciais são inadequadas" para o DEMON.
* **Comparação com MEL/LOFAR:** o DEMON isolado fica abaixo do MLP LOFAR do artigo (66.51%) em MLP 14.7, SVM m=6000 21.6, RF 22.1, CNN 30.2 p.p. e abaixo do SVM MEL local (63.78%) em MLP 12.0, SVM m=6000 18.8, RF 19.3, CNN 27.5 p.p. Isso indica que o DEMON sozinho é uma representação mais fraca nestes modelos; não "comprova" um teto teórico. Detalhes no Notebook 8, em `reports/decisao_metodo_demon.md` e em `reports/comparacao_modelos_demon.md`.

### 5.7 Fusão Multimodal Direta (MEL + DEMON)
A combinação do MEL (256 filtros, cache de 5,12 s) com o DEMON (253 bins) foi testada no protocolo $5\times2$cv (particionamento por gravação, 1706 áudios):
* **MLP Fusão** (`mlp_mel5s-demon_h64-32_e35_4c`, rede [64,32], **maior** que a [32,16] das entradas isoladas): SP **66.92 ± 1.59%**, ACC **67.40 ± 1.58%**. Contra o MLP LOFAR local (SP 66.24 ± 1.78): p = 0.676, **sem diferença significativa**; contra o MLP MEL 5,12 s: p = 0.358. O MLP LOFAR do artigo é 66.51 ± 1.39% e a CNN LOFAR do artigo 66.05 ± 1.90%; como esses valores não têm folds em disco, **não** é possível afirmar que a fusão os "supera" nem que estabelece "recorde".
* **Recall de SMALL:** MLP MEL 5,12 s: 46.07% (4 classes) / 54.04% (3 classes); MLP Fusão: 53.03% / 59.04% (+6.97 p.p. / +5.00 p.p.; F-test do recall SMALL 4c: p = 0.574). BACKGROUND na fusão: 81.84%; LARGE: 70.89% (4c) / 71.50% (3c).
* **Desvio entre folds:** SP do MLP MEL 5,12 s ±2.19 vs. ±1.59 na fusão. Uma diferença de desvio em 10 folds não "comprova" complementaridade ortogonal.

---

## 6. Conclusões e Delineamento Operacional

Com os números atuais, o desenho em duas camadas continua sendo uma **proposta operacional**, não um resultado estatisticamente demonstrado:

1. **Camada de cobertura (SVM MEL com $t \ge 0.6$):** aceita 76.07 ± 2.69% dos arquivos com ACC balanceada de 69.25 ± 1.68% (MLP MEL no mesmo limiar: cobertura 70.36 ± 2.47%, ACC 71.27 ± 1.84%).
2. **Camada de alta certeza ($t \ge 0.9$):** SVM LOFAR (C=0.5, PCA64) com ACC balanceada de 81.55 ± 4.30% sobre 25.10 ± 1.34% dos arquivos; MLP MEL com 75.75 ± 4.06% sobre 32.90 ± 2.17%; MLP LOFAR com 81.20 ± 3.78% sobre 24.84 ± 2.44%. A afirmação anterior de "25,6% de falsos alarmes" da CNN e de "alarmes falsos catastróficos" não tem base nestes dados (ver 5.3).
3. **Sem rejeição**, nenhum dos SVM MEL/LOFAR/Híbrido, a fusão MLP ou as cascatas re-executadas tem SP significativamente diferente do MLP LOFAR local (maior SP sem vazamento entre as cascatas por janela: 67.16 ± 1.82, p = 0.563; menor p desse conjunto: p = 0.146). O melhor baseline do artigo (MLP LOFAR, 66.51 ± 1.39) não foi superado com significância.

---

## 7. Robustez ao CPA e Experimentos de Proximidade (Tabela 10 do Artigo)

Replicou-se o experimento de **CPA Proximity** da Tabela 10 do artigo (*Silva et al., 2025*):
* **Dataset A (Near CPA):** trechos próximos do ponto de máxima aproximação (maior SNR).
* **Dataset C (Far CPA):** trechos com a embarcação distante (menor SNR).

1. **Trained on A:** treino em A, avaliação em A e C.
2. **Trained on C:** treino em C, avaliação em A e C.

### Tabela 3: Desempenho no Experimento de CPA Proximity (re-execução local, `results/trainings/cpa_exp`)

| Modelo / Classificador | Treinado em | SP A (%) | ACC A (%) | SP C (%) | ACC C (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Forest Mel** | Dataset A (Near) | 57.05 ± 3.48 | 59.24 ± 3.15 | 48.22 ± 5.61 | 51.87 ± 4.97 |
| **Forest Mel** | Dataset C (Far) | 48.84 ± 4.53 | 51.93 ± 2.88 | 45.00 ± 6.37 | 50.07 ± 4.41 |
| **MLP Mel** | Dataset A (Near) | 67.33 ± 2.67 | 67.74 ± 2.69 | 60.16 ± 5.99 | 61.03 ± 5.18 |
| **MLP Mel** | Dataset C (Far) | 59.36 ± 4.55 | 59.70 ± 4.47 | 59.46 ± 4.45 | 60.21 ± 4.17 |
| **CNN Mel** | Dataset A (Near) | 61.84 ± 3.26 | 62.61 ± 2.85 | 56.58 ± 4.80 | 58.09 ± 4.30 |
| **CNN Mel** | Dataset C (Far) | 52.41 ± 7.12 | 53.28 ± 6.63 | 55.37 ± 5.29 | 56.40 ± 4.81 |
| **SVM Mel (m=4000, C=2.0)** | Dataset A (Near) | 64.15 ± 2.87 | 65.08 ± 2.91 | 59.24 ± 4.55 | 60.84 ± 3.59 |
| **SVM Mel (m=4000, C=2.0)** | Dataset C (Far) | 56.85 ± 4.89 | 57.64 ± 4.33 | 59.13 ± 6.03 | 60.46 ± 5.04 |
| **SVM LOFAR (m=4000, C=2.0, PCA64)** | Dataset A (Near) | 67.24 ± 3.42 | 68.23 ± 2.94 | 55.35 ± 4.45 | 57.26 ± 3.66 |
| **SVM LOFAR (m=4000, C=2.0, PCA64)** | Dataset C (Far) | 58.52 ± 4.95 | 59.59 ± 4.65 | 55.84 ± 6.00 | 57.38 ± 5.49 |

*Todas as linhas são re-execuções locais (Forest/MLP/CNN com os modelos do artigo; não são os valores publicados na Tabela 10). ACC = acurácia balanceada. O problema tem 3 classes (acaso da ACC balanceada = 33.3%).*

### Conclusões e Análise das Métricas de Proximidade:

1. **Queda de ACC ao passar de A para C (treino em A):** Forest Mel: 7.37 p.p.; MLP Mel: 6.71 p.p.; CNN Mel: 4.52 p.p.; SVM Mel (m=4000, C=2.0): 4.25 p.p.; SVM LOFAR (m=4000, C=2.0, PCA64): 10.97 p.p.. A menor queda é a do **SVM Mel (m=4000, C=2.0)** e a maior a do **SVM LOFAR (m=4000, C=2.0, PCA64)**. A diferença de queda entre o SVM Mel e o MLP Mel tem p = 0.695 (**sem diferença significativa**); a afirmação anterior de que o SVM "deforma-se de maneira muito mais resiliente" não é sustentada como diferença estatística.
2. **Treino em C (Far):** a CNN Mel atinge 56.40 ± 4.81% em C e 53.28 ± 6.63% em A; o SVM Mel atinge 60.46 ± 5.04% em C e 57.64 ± 4.33% em A (F-test do ACC: p = 0.322 em C, p = 0.277 em A). A frase anterior de que 53.28% estaria "próximo do limiar aleatório" era exagerada: o acaso da ACC balanceada é 33.3% (3 classes), e a CNN fica 19.9 p.p. acima dele. A vantagem antiga "de mais de 4% em ambas as frentes" do SVM sobre a CNN não é significativa.
3. **Desvio entre folds no Dataset C (ACC):** Forest Mel: 4.97 (treino A) / 4.41 (treino C); MLP Mel: 5.18 (treino A) / 4.17 (treino C); CNN Mel: 4.30 (treino A) / 4.81 (treino C); SVM Mel (m=4000, C=2.0): 3.59 (treino A) / 5.04 (treino C); SVM LOFAR (m=4000, C=2.0, PCA64): 3.66 (treino A) / 5.49 (treino C).
4. **SVM LOFAR:** em A→A, o SVM LOFAR tem ACC 68.23 ± 2.94% contra 67.74 ± 2.69% do MLP Mel (p = 0.589); em SP, 67.24 ± 3.42% contra 67.33 ± 2.67% (p = 0.628). O modelo com maior ACC A→A é o SVM LOFAR (m=4000, C=2.0, PCA64), mas sem diferença significativa para o MLP; o "recorde absoluto" anterior não é sustentado. Em C→A, o SVM LOFAR tem 59.59 ± 4.65% contra 53.28 ± 6.63% da CNN Mel (p = 0.015).

---

## 8. Landmarks via MiniBatchKMeans

A proposta de substituir a amostragem aleatória de landmarks do Nyström por centroides de `MiniBatchKMeans` já tem experimentos em disco (MEL, m=4000):

| Variante | Experimento | SP (%) | F-test vs. SVM MEL aleatório (SP 63.78 ± 1.14) |
| :--- | :--- | :---: | :---: |
| MEL m=4000, landmarks k-means | `svm_nystroem_4000_log_melgram_kmeans_elasticnet_l1r0.15` | 63.56 ± 2.40 | p = 0.446 |
| MEL m=4000, k-means estratificado balanceado | `svm_nystroem_4000_log_melgram_stratified_kmeans_balanced_elasticnet_l1r0.15` | incompleto (< 10 folds) | — |

As expectativas listadas antes (mesma acurácia com $m$ menor, melhor cobertura de `SMALL`) eram hipóteses sem referência bibliográfica nem experimento; ficam como hipóteses.

---

## 9. Arquitetura em Cascata (Duplo Especialista)

Sistema em dois estágios:

1. **Estágio 1 (multiclasse):** SVM Nyström em 4 classes — Híbrido MEL+LOFAR (m=4000, C=2.0, norm; recall de LARGE 74.8%), MEL (m=4000) ou LOFAR (m=6000), conforme a variante.
2. **Estágio 2 (especialistas binários m=6000):**
   * predições `SMALL`/`MEDIUM` → **Especialista SM** (MEL-256 + LOFAR-PCA64; `svm_nystroem_6000_hybrid_binary_small_medium_pca64_elasticnet_l1r0.15`);
   * predições `LARGE` → **Especialista SL** (MEL-256 + LOFAR-PCA8; `svm_nystroem_6000_hybrid_binary_small_large_pca8_elasticnet_l1r0.15`).
   * Roteamento "runner-up": o SL só é acionado quando a 2ª classe mais votada é SMALL.

> ⚠️ **Vazamento corrigido.** O especialista SL de junho/2026 usava um `StratifiedKFold` só sobre {SMALL, LARGE}: **1925/3920 = 49.1%** dos arquivos SMALL/LARGE de teste estavam no **treino** do especialista do mesmo fold. Ele foi **retreinado em 20/10/2026** (agora 0/3920) e todas as cascatas que o usam foram re-executadas. As versões antigas estão em `results/trainings/_cascatas_com_sl_vazado_pre20261020/` e aparecem só na coluna "SP antes (SL vazado)". A cascata `..._smallalso_large_svm_nystroem_6000_hybrid_binary_small_large_pca8_elasticnet_l1r0.15` (sem `fbmc`, versão antiga do `svm.py`, SP antes 67.57 ± 1.78) **não foi re-executada e está invalidada**. O especialista SM não tem vazamento.

#### Tabela 4: Cascatas (recalls, ACC balanceada e SP; p = F-test 5×2cv do SP)

| Variante | Recall SMALL | Recall MEDIUM | Recall LARGE | Recall BG | ACC | SP | SP antes (SL vazado) | p vs. estágio 1 | p vs. MLP LOFAR local |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Estágio 1: MEL (m=4000, C=2.0) | 50.7 ± 10.5 | 60.4 ± 11.6 | 69.3 ± 2.8 | 77.9 ± 6.1 | 64.6 ± 1.2 | 63.78 ± 1.14 | — | — | p = 0.266 |
| Estágio 1: Híbrido (m=4000, C=2.0, norm) | 51.1 ± 10.1 | 62.3 ± 9.3 | 74.8 ± 3.9 | 73.1 ± 3.6 | 65.3 ± 1.9 | 64.67 ± 2.24 | — | — | p = 0.338 |
| Estágio 1: LOFAR (m=6000) | 38.1 ± 5.9 | 60.4 ± 7.9 | 76.3 ± 4.4 | 87.0 ± 4.5 | 65.5 ± 1.8 | 63.83 ± 1.91 | — | — | p = 0.146 |
| MLP LOFAR (local, referência) | 46.3 ± 6.1 | 61.5 ± 4.4 | 74.7 ± 4.3 | 86.3 ± 3.5 | 67.2 ± 1.6 | 66.24 ± 1.78 | — | — | — |
| *Cascatas com o especialista SMALL×LARGE retreinado (20/10/2026)* | | | | | | | | | |
| HYBRID-MC(C=2.0) → SM(PCA64)+SL(PCA8) — sempre (por janela) | 59.4 ± 4.8 | 61.6 ± 5.4 | 69.5 ± 2.5 | 73.3 ± 3.5 | 66.0 ± 1.2 | 65.77 ± 1.23 | 68.07 ± 1.65 | p = 0.028 | p = 0.573 |
| HYBRID-MC(C=2.0) → SM(PCA64)+SL(PCA8) — runner-up (por janela) | 58.0 ± 4.5 | 62.0 ± 5.1 | 71.1 ± 2.5 | 73.3 ± 3.5 | 66.1 ± 1.4 | 65.87 ± 1.41 | 67.59 ± 1.95 | p = 0.035 | p = 0.566 |
| LOFAR-MC(m=6000) → SL(PCA8) (por janela) | 63.7 ± 5.3 | 57.7 ± 5.8 | 63.5 ± 3.5 | 85.7 ± 4.6 | 67.6 ± 1.9 | 67.16 ± 1.82 | 69.46 ± 2.25 | p = 0.007 | p = 0.563 |
| LOFAR-MC(m=6000) → SM(PCA64) só MEDIUM + SL(PCA8) (por janela) | 71.7 ± 3.8 | 43.8 ± 3.9 | 62.7 ± 2.9 | 85.5 ± 4.6 | 65.9 ± 1.8 | 64.93 ± 1.81 | incompleto | p = 0.254 | p = 0.416 |
| LOFAR-MC(m=6000) → SL(PCA8) (por arquivo, `eval_cascade_by_ship.py`) | 51.7 ± 7.1 | 60.4 ± 7.9 | 68.0 ± 3.1 | 86.9 ± 4.4 | 66.8 ± 1.9 | 66.00 ± 1.90 | 68.56 ± 1.67 | p = 0.030 | p = 0.371 |
| *Cascatas que não usam o especialista SMALL×LARGE (válidas; estágio 1 não registrado no config)* | | | | | | | | | |
| MEL → SM(PCA64)/LOFAR (SMALL+MEDIUM) | 44.8 ± 7.4 | 62.0 ± 8.8 | 70.7 ± 5.2 | 80.0 ± 3.5 | 64.4 ± 1.4 | 63.47 ± 1.39 | — | — | p = 0.174 |
| HÍBRIDO → SM(PCA64)/LOFAR (só MEDIUM) | 63.9 ± 6.2 | 55.4 ± 4.1 | 62.2 ± 4.3 | 80.5 ± 3.7 | 65.5 ± 2.0 | 65.13 ± 1.98 | — | — | p = 0.353 |
| HÍBRIDO → SM(PCA64)/LOFAR (SMALL+MEDIUM) | 58.1 ± 7.7 | 61.6 ± 5.0 | 62.4 ± 4.4 | 80.7 ± 3.7 | 65.7 ± 1.9 | 65.33 ± 1.91 | — | — | p = 0.202 |
| MEL → cascata 1000 (k-means) | 45.1 ± 8.5 | 61.8 ± 8.0 | 69.3 ± 6.1 | 80.7 ± 6.6 | 64.2 ± 2.0 | 63.24 ± 2.24 | — | — | p = 0.065 |
| Cascata dupla com especialista LOFAR (t=0.55) | 61.7 ± 5.8 | 51.7 ± 7.3 | 65.4 ± 6.1 | 80.8 ± 5.1 | 64.9 ± 1.9 | 64.33 ± 2.12 | — | — | p = 0.458 |
| ~~MEL→SM(PCA64)+SL(PCA8) (SMALL+MEDIUM)~~ **INVALIDADA** | — | — | — | — | — | — | 67.57 ± 1.78 | — | — |

### Discussão dos Resultados da Cascata:
1. **Sem vazamento, os ganhos encolhem:** a variante "sempre" com estágio 1 híbrido caiu de 68.07 ± 1.65 para **65.77 ± 1.23**; a "runner-up", de 67.59 ± 1.95 para **65.87 ± 1.41**; a cascata por arquivo, de 68.56 ± 1.67 para **66.00 ± 1.90**.
2. **Melhor cascata por janela:** LOFAR-MC(m=6000) → SL(PCA8), SP **67.16 ± 1.82** (recall SMALL 63.7%, LARGE 63.5%). Contra o próprio estágio 1: p = 0.007 (diferença **significativa**); contra o MLP LOFAR local (66.24 ± 1.78): p = 0.563.
3. **Nenhuma** cascata re-executada difere significativamente do MLP LOFAR local (menor p: p = 0.371); os SP vão de 64.93 a 67.16, contra 66.51 ± 1.39 do MLP LOFAR do artigo. A "melhor" cascata foi escolhida olhando o teste (entre 4 variantes), o que torna otimista sua vantagem sobre o estágio 1. As afirmações anteriores "melhor arquitetura geral" (68,05%) e "empata com a cascata MEL pura (67,59 vs 67,58)" vinham do especialista vazado e foram removidas.
4. **"Sempre" vs. "runner-up":** com o SL retreinado, "sempre" tem SMALL 59.4% e LARGE 69.5%; "runner-up" tem SMALL 58.0% e LARGE 71.1%.
5. **SP fold-wise:** $SP = \sqrt{\bar{r} \cdot \tilde{r}} \times 100$ ($\bar{r}$ = média aritmética dos recalls, $\tilde{r}$ = média geométrica), calculado por fold e reportado como média ± desvio (ddof=0) sobre os 10 folds.

---

## 10. Fusão Multimodal Direta (MEL + DEMON) entre Famílias de Modelos

Avaliou-se o espaço de **509 dimensões** (256 Mel + 253 DEMON, janelas de 5,12 s sincronizadas) nas famílias do benchmark, sob o protocolo $5\times2$cv com particionamento por gravação (`exclusive_header='ID'`), mesma grade do Notebook 12.

### Tabela 5: Classificadores na Fusão Multimodal (5x2cv)

| Arquitetura / Modelo | Representação | ACC 4C (%) | SP 4C (%) | ACC 3C (%) | SP 3C (%) | Recall SMALL (4C / 3C) | Experimento |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| MLP (PyTorch) | DEMON isolado | 55.08 ± 1.75 | 53.09 ± 2.69 | 57.22 ± 1.02 | 56.51 ± 1.33 | 31.85% / 65.62% | `mlp_demon_h32-16_e35_4c` ([32,16], 35 ép., 5,12 s) |
| MLP (PyTorch) | MEL isolado | 64.96 ± 1.88 | 64.16 ± 2.19 | 64.19 ± 1.76 | 63.81 ± 1.81 | 46.07% / 54.04% | `mlp_mel5s_h32-16_e35_4c` ([32,16], 35 ép., MEL 5,12 s) |
| MLP (PyTorch) | LOFAR isolado (local) | 67.22 ± 1.65 | 66.24 ± 1.78 | — | — | 46.35% / — | `mlp_lofar` (pipeline iara, 0,512 s) |
| **MLP (PyTorch)** | **FUSÃO MULTIMODAL** | 67.40 ± 1.58 | 66.92 ± 1.59 | 65.26 ± 1.56 | 65.06 ± 1.61 | 53.03% / 59.04% | `mlp_mel5s-demon_h64-32_e35_4c` ([64,32] (rede maior), 35 ép., 5,12 s) |
| *---* | *---* | *---* | *---* | *---* | *---* | *---* | *---* |
| SVM Nyström | DEMON isolado | 52.73 ± 1.86 | 44.95 ± 4.58 | 55.55 ± 2.55 | 53.82 ± 3.33 | 11.85% / 50.22% | `svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000_4c` (m=6000) |
| SVM Nyström | MEL isolado | 64.56 ± 1.18 | 63.78 ± 1.14 | — | — | 50.73% / — | `svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0` (m=4000 (não 6000), 0,512 s) |
| SVM Nyström | LOFAR isolado (local) | 65.47 ± 1.77 | 63.83 ± 1.91 | — | — | 38.15% / — | `svm_nystroem_6000_lofar_elasticnet_l1r0.15` (m=6000, 0,512 s) |
| **SVM Nyström** | **FUSÃO MULTIMODAL** | 65.60 ± 1.27 | 64.67 ± 1.41 | 63.45 ± 1.49 | 62.92 ± 1.55 | 46.24% / 57.53% | `svm_nystroem_6000_mel5s-demon_C2.0_elasticnet_l1r0.15_4c` (m=6000, 5,12 s) |
| *---* | *---* | *---* | *---* | *---* | *---* | *---* | *---* |
| Random Forest (100 árv.) | DEMON isolado | 46.08 ± 1.79 | 44.44 ± 2.24 | 54.93 ± 1.71 | 53.73 ± 1.68 | 52.47% / 60.06% | `rf_demon_n100_balanced_4c` (100 árv.) |
| Random Forest (100 árv.) | MEL isolado | 61.61 ± 1.88 | 61.50 ± 1.89 | — | — | 57.87% / — | `rf_mel5s_n100_balanced_4c` (100 árv., MEL 5,12 s) |
| Random Forest (100 árv.) | LOFAR isolado (local) | 63.79 ± 1.70 | 63.07 ± 1.81 | — | — | 47.81% / — | `rf_lofar5s_n100_balanced_4c` (100 árv., LOFAR 5,12 s) |
| **Random Forest (100 árv.)** | **FUSÃO MULTIMODAL** | 62.15 ± 1.67 | 62.04 ± 1.66 | 61.27 ± 2.09 | 61.17 ± 2.12 | 59.16% / 59.49% | `rf_mel5s-demon_n100_balanced_4c` (100 árv., 5,12 s) |
| *---* | *---* | *---* | *---* | *---* | *---* | *---* | *---* |
| CNN 2D | DEMON isolado | 44.48 ± 2.03 | 36.28 ± 6.38 | 44.15 ± 1.29 | 37.06 ± 12.48 | 8.71% / 21.46% | `cnn2d_demon_w32_b32_e20_4c` (CNN ad hoc [COLAPSADA]) |
| CNN 2D | MEL isolado | 65.01 ± 2.05 | 63.80 ± 2.23 | — | — | 40.90% / — | `cnn` (CNN oficial, 0,512 s) |
| CNN 2D | LOFAR isolado (local) | 47.58 ± 2.43 | 35.40 ± 12.38 | — | — | 39.21% / — | `cnn2d_lofar5s_w32_b64_e15_4c` (CNN ad hoc, LOFAR 5,12 s [COLAPSADA]) |
| **CNN 2D** | **FUSÃO MULTIMODAL** | 39.57 ± 10.26 | 8.60 ± 14.13 | 43.16 ± 2.55 | 31.46 ± 17.01 | 13.82% / 37.13% | `cnn2d_mel5s-demon_w32_b32_e25_4c` (CNN ad hoc [COLAPSADA]) |

*"3C" = modelo **retreinado** sem BACKGROUND (`*_3c`). "—" = não há experimento 3C em disco para essa linha (os valores 3C que a versão anterior trazia para MEL/LOFAR — p.ex. 65,50 / 64,20 no MLP LOFAR — não correspondiam a nenhum experimento e foram removidos). As linhas "LOFAR isolado" agora são re-execuções locais (não a Tabela 7 do artigo). O RF e a CNN LOFAR usam o cache LOFAR de 5,12 s; a CNN LOFAR/DEMON/FUSÃO ad hoc está colapsada.*

### 10.1 A fusão MEL + DEMON empata com o LOFAR?

1. **F-test 5×2cv (SP), FUSÃO vs. LOFAR isolado:** MLP p = 0.676; SVM p = 0.177; RF p = 0.393. No RF, o LOFAR local (SP 63.07 ± 1.81) é maior que a fusão (SP 62.04 ± 1.66). Os modelos são, portanto, comparáveis, mas a afirmação de que a fusão "supera" o LOFAR não se sustenta. A interpretação de que MEL + DEMON "recompõe exatamente os mesmos fenômenos físicos que o LOFAR" é uma hipótese, não algo medido aqui. (O MEL e o LOFAR cobrem 0–8 kHz após a reamostragem a 16 kHz; o DEMON cobre 0,39–25 Hz.)
2. **Recall de SMALL:** no LOFAR isolado (MLP/SVM/RF) fica entre 38.1% e 47.8%; na fusão é 53.03% (MLP), 46.24% (SVM, 4C) / 57.53% (SVM, 3C) e 59.16% (RF). O maior recall de SMALL na fusão em 4C é o do RF. A explicação pela frequência de pás ("15 a 25 Hz") é uma hipótese.
3. **CNN 2D:** a CNN ad hoc na fusão tem SP 8.60 ± 14.13 — colapso de treino (folds com recall nulo). A hipótese de que a descontinuidade entre as bandas Mel e DEMON prejudica as convoluções não foi testada.
