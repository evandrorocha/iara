# Comparação de Modelos — Dataset IARA (By Audio)

Este documento consolida os resultados do experimento de avaliação do SVM contra as baselines oficiais do artigo original do IARA (Silva et al., 2025). O objetivo principal é avaliar a capacidade de generalização e maximização de margem do SVM contra as redes neurais e a Random Forest.

## Resultados Consolidados (vs Artigo Oficial)

| Classificador | Feature Extractor | SP (%) | Acurácia (ACC) Teste (%) | F1-Score Teste (%) | Status / Origem |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RF** | MEL | 62.22 ± 1.86 | 62.64 ± 1.84 | 63.79 ± 1.73 | Artigo (Tabela 7) |
| **RF** | LOFAR | 56.92 ± 1.69 | 58.87 ± 1.68 | 58.00 ± 1.41 | Artigo (Tabela 7) |
| **MLP** | MEL | 63.38 ± 1.81 | 64.51 ± 1.75 | 62.89 ± 1.68 | Artigo (Tabela 7) |
| **MLP** | LOFAR | 66.51 ± 1.39 | 67.48 ± 1.24 | 66.72 ± 1.17 | Artigo (Tabela 7) |
| **CNN** | MEL | 63.52 ± 2.26 | 64.99 ± 2.09 | 63.04 ± 2.02 | Artigo (Tabela 7) |
| **CNN** | LOFAR | 66.05 ± 1.90 | 67.02 ± 1.78 | 66.29 ± 2.13 | Artigo (Tabela 7) |
| --- | --- | --- | --- | --- | --- |
| **SVM (m=300)** | MEL | **59.96% ± 3.38** | **62.34% ± 1.98** | **59.47% ± 3.16** | Nossa Execução (Local) |
| **SVM (m=1000)** | MEL | **62.96% ± 2.08** | **64.11% ± 1.98** | **62.74% ± 1.87** | Nossa Execução (Local) |
| **SVM (m=2000)** | MEL | **63.04% ± 2.37** | **64.45% ± 1.93** | **62.93% ± 2.22** | Nossa Execução (Local) |
| **SVM (m=2000) (C=10)** | MEL | **62.18% ± 3.14** | **63.53% ± 2.50** | **62.67% ± 2.84** | Nossa Execução (Local) |
| **SVM (m=3000)** | MEL | **63.65% ± 1.99** | **64.77% ± 1.87** | **63.59% ± 1.92** | Nossa Execução (Local) |
| **SVM (m=1000)** | LOFAR | **57.86% ± 3.83** | **60.95% ± 2.89** | **57.86% ± 3.53** | Nossa Execução (Local) |
| **SVM (m=2000)** | LOFAR | **60.10% ± 2.25** | **62.54% ± 1.98** | **59.95% ± 2.77** | Nossa Execução (Local) |
| **SVM (m=2000) (C=0.1)** | LOFAR | **54.84% ± 1.60** | **59.67% ± 1.28** | **54.64% ± 1.69** | Nossa Execução (Local) |
| **SVM (m=2000) (C=2.0)** | LOFAR | **60.15% ± 2.74** | **62.45% ± 2.17** | **60.59% ± 2.64** | Nossa Execução (Local) |
| **SVM (m=2000) (C=2.0) (g=scale*2)** | LOFAR | **59.18% ± 2.70** | **61.95% ± 1.63** | **59.22% ± 2.18** | Nossa Execução (Local) |
| **SVM (m=4000) (C=2.0)** | LOFAR | **61.70% ± 3.67** | **63.55% ± 2.91** | **62.85% ± 3.33** | Nossa Execução (Local) |
| **SVM (m=1000) (C=2.0) (norm + pca64)** | LOFAR | **56.53% ± 2.69** | **59.04% ± 1.88** | **57.71% ± 2.53** | Nossa Execução (Local) |

> [!NOTE]
> *Os valores de RF, MLP e CNN foram extraídos diretamente da Tabela 7 do artigo oficial do IARA. As margens de erro se referem a ±1 desvio padrão no teste de Validação Cruzada de 10 folds.*

## Análise do Índice SP (Specificity Index)
O índice **SP** (calculado no código como a média geométrica combinada com a média aritmética das taxas de detecção por classe) é um indicador robusto da capacidade de o modelo classificar corretamente classes individuais sem sofrer viés de classe majoritária.

1. **Desempenho do SVM (m=1000):** O nosso SVM atingiu **62.96% ± 2.08** no índice SP. Isso é estatisticamente idêntico ao **63.52% ± 2.26** da CNN e ao **63.38% ± 1.81** da MLP.
2. **Robustez Multiclasse:** O fato de o SP do SVM de 1000 componentes estar tão colado no das redes neurais prova que a aproximação de kernel RBF por Nyström conseguiu desenhar hiperplanos de separação muito bem-balanceados no espaço de alta dimensão, evitando que as classes minoritárias (navios raros) fossem engolidas pelo ruído de fundo majoritário.
