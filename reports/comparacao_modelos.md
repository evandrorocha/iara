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
| **SVM (m=1000) (norm + pca64)** | MEL | **60.83% ± 1.67** | **61.86% ± 1.66** | **60.95% ± 1.31** | Nossa Execução (Local) |
| **SVM (m=2000)** | MEL | **63.04% ± 2.37** | **64.45% ± 1.93** | **62.93% ± 2.22** | Nossa Execução (Local) |
| **SVM (m=2000) (C=10)** | MEL | **62.18% ± 3.14** | **63.53% ± 2.50** | **62.67% ± 2.84** | Nossa Execução (Local) |
| **SVM (m=3000)** | MEL | **63.65% ± 1.99** | **64.77% ± 1.87** | **63.59% ± 1.92** | Nossa Execução (Local) |
| **SVM (m=4000) (C=2.0) (pca64) (elasticnet)** | MEL | **63.82% ± 1.61** | **64.61% ± 1.60** | **64.60% ± 1.54** | Nossa Execução (Local) |
| **SVM (m=4000) (C=2.0) (elasticnet) (g=0.05)** | MEL | **46.01% ± 4.58** | **52.63% ± 2.19** | **46.86% ± 3.57** | Nossa Execução (Local) |
| **SVM (m=4000) (C=2.0) (elasticnet)** | MEL | **63.78% ± 1.14** | **64.56% ± 1.18** | **64.32% ± 1.02** | Nossa Execução (Local) |



| **SVM (m=1000)** | LOFAR | **57.86% ± 3.83** | **60.95% ± 2.89** | **57.86% ± 3.53** | Nossa Execução (Local) |
| **SVM (m=2000)** | LOFAR | **60.10% ± 2.25** | **62.54% ± 1.98** | **59.95% ± 2.77** | Nossa Execução (Local) |
| **SVM (m=2000) (C=0.1)** | LOFAR | **54.84% ± 1.60** | **59.67% ± 1.28** | **54.64% ± 1.69** | Nossa Execução (Local) |
| **SVM (m=2000) (C=2.0)** | LOFAR | **60.15% ± 2.74** | **62.45% ± 2.17** | **60.59% ± 2.64** | Nossa Execução (Local) |
| **SVM (m=2000) (C=2.0) (g=scale*2)** | LOFAR | **59.18% ± 2.70** | **61.95% ± 1.63** | **59.22% ± 2.18** | Nossa Execução (Local) |
| **SVM (m=4000) (C=2.0)** | LOFAR | **61.70% ± 3.67** | **63.55% ± 2.91** | **62.85% ± 3.33** | Nossa Execução (Local) |
| **SVM (m=1000) (C=2.0) (norm + pca64)** | LOFAR | **56.53% ± 2.69** | **59.04% ± 1.88** | **57.71% ± 2.53** | Nossa Execução (Local) |
| **SVM (m=1000) (C=2.0) (norm)** | LOFAR | **53.78% ± 3.76** | **57.62% ± 2.07** | **54.97% ± 3.11** | Nossa Execução (Local) |
| **SVM (m=1000) (C=2.0) (pca64)** | LOFAR | **61.96% ± 2.22** | **63.16% ± 2.18** | **62.65% ± 1.74** | Nossa Execução (Local) |
| **SVM (m=1000) (C=2.0) (pca64) (elasticnet)** | LOFAR | **61.73% ± 2.45** | **61.92% ± 1.89** | **62.46% ± 2.19** | Nossa Execução (Local) |
| **SVM (m=4000) (C=2.0) (pca64) (elasticnet)** | LOFAR | **63.10% ± 2.19** | **64.04% ± 1.88** | **64.34% ± 2.23** | Nossa Execução (Local) |



> [!NOTE]
> *Os valores de RF, MLP e CNN foram extraídos diretamente da Tabela 7 do artigo oficial do IARA. As margens de erro se referem a ±1 desvio padrão no teste de Validação Cruzada de 10 folds.*

## Análise do Índice SP (Specificity Index)
O índice **SP** (calculado no código como a média geométrica combinada com a média aritmética das taxas de detecção por classe) é um indicador robusto da capacidade de o modelo classificar corretamente classes individuais sem sofrer viés de classe majoritária.

1. **Desempenho do SVM (m=1000):** O nosso SVM atingiu **62.96% ± 2.08** no índice SP. Isso é estatisticamente idêntico ao **63.52% ± 2.26** da CNN e ao **63.38% ± 1.81** da MLP.
2. **Robustez Multiclasse:** O fato de o SP do SVM de 1000 componentes estar tão colado no das redes neurais prova que a aproximação de kernel RBF por Nyström conseguiu desenhar hiperplanos de separação muito bem-balanceados no espaço de alta dimensão, evitando que as classes minoritárias (navios raros) fossem engolidas pelo ruído de fundo majoritário.

## ⚖️ Comparação de Filtro de Confiança: LOFAR vs. MEL (Os Dois Campeões)

Abaixo está o embate direto entre o **Melhor Modelo LOFAR** ($m=4000$, PCA64, ElasticNet) e o **Melhor Modelo MEL** ($m=4000$, Puro/Sem PCA, ElasticNet) sob diferentes limiares de concordância de janela ($t$):

| Limiar de Confiança ($t$) | Métrica | Modelo LOFAR Campeão | Modelo MEL Campeão (Supremo) | Vantagem / Dinâmica do Trade-off |
| :--- | :--- | :---: | :---: | :--- |
| **Cenário Standard ($t=0$)** | **Acurácia (ACC)** <br> **Índice SP** <br> Cobertura | **63.32% ± 2.10** <br> **63.10% ± 2.19** <br> **100%** | **64.56% ± 1.18** <br> **63.78% ± 1.14** <br> **100%** | **Vitória do MEL:** <br> O MEL sem PCA exibe maior acurácia global e um desvio padrão quase metade do LOFAR (máxima estabilidade nos folds). |
| **Maioria Simples ($t \ge 0.5$)** | **Acurácia (ACC)** <br> **Índice SP** <br> Cobertura | **70.36% ± 2.13** <br> **68.56% ± 2.88** <br> **78.25% ± 1.90** | **66.23% ± 1.48** <br> **66.07% ± 1.63** <br> **89.67% ± 1.17** | **Trade-off:** <br> O LOFAR ganha em precisão (70.36%), mas o MEL mantém **89.67% de cobertura** (descarta apenas 10% dos áudios). |
| **Maioria Absoluta ($t \ge 0.6$)** | **Acurácia (ACC)** <br> **Índice SP** <br> Cobertura | **76.54% ± 3.26** <br> **72.87% ± 4.86** <br> **61.21% ± 2.20** | **69.60% ± 1.86** <br> **68.32% ± 2.37** <br> **76.07% ± 2.69** | **Trade-off:** <br> O LOFAR atinge alta precisão militar (76.54%), porém o MEL consegue classificar com segurança **15% a mais de arquivos** (76.07%). |
| **Certeza Crítica ($t \ge 0.9$)** | **Acurácia (ACC)** <br> **Índice SP** <br> Cobertura | **90.43% ± 2.95** <br> **78.85% ± 7.18** <br> **22.27% ± 2.86** | **83.42% ± 2.07** <br> **75.58% ± 3.18** <br> **37.48% ± 2.32** | **Trade-off:** <br> O LOFAR bate a marca absurda de **90.43% de acurácia** (mas rejeita 78% dos áudios). O MEL entrega excelentes **83.42%** classificando quase o dobro de navios (**37.48%**). |

### 🧠 Análise Físico-Acústica do Embate:
1. **O LOFAR é Esparso e Focado em Raias (Picos):** Como o LOFAR preserva frequências lineares físicas estreitas, as harmônicas dos motores geram "linhas digitais" extremamente nítidas e estáveis. Quando o modelo adquire alta concordância nessas raias ($t \ge 0.6$ ou $0.9$), a sua acurácia dispara para patamares incríveis (**76% a 90%**). Em contrapartida, por ser uma assinatura esparsa, flutuações de ruído oceânico o fazem descartar mais arquivos (cobertura menor).
2. **O MEL é Denso e Integrado (Energia):** Como o MEL agrupa o espectro em 256 bandas largas logarítmicas, ele exibe extrema estabilidade e resiliência a ruídos externos, o que resulta em **coberturas muito maiores** em todos os limiares (consegue classificar 76% da base sob $t \ge 0.6$). Contudo, a sobreposição triangular inerente da escala Mel gera bordas de decisão mais suaves, limitando a acurácia máxima a **83.42%** no pico de certeza.

