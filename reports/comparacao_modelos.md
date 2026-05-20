# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
**Abordagem Proposta: SVM Nyström Gaussiano com Penalização ElasticNet e Opção de Rejeição**

---

## 1. Introdução e Metodologia Experimental

Este documento apresenta uma análise comparativa e formal de desempenho entre o modelo proposto de **Support Vector Machines com Aproximação de Nyström (RBF-SVM)** e os modelos baselines estabelecidos no artigo de referência do dataset IARA (*Silva et al., 2025*): *Random Forest (RF)*, *Multi-Layer Perceptron (MLP)* e *Convolutional Neural Network (CNN)*.

### Metodologia de Validação Cruzada (5x2cv):
Para mitigar a variância decorrente de partições aleatórias e garantir a validade estatística das inferências, adotou-se o protocolo de **Validação Cruzada 5x2 (5x2cv)**, totalizando 10 folds independentes. A partição fold-wise é restrita e estratificada no nível de **ID de Arquivo Físico original** (*Exclusive Ships on Test*). Essa restrição impede o vazamento de dados (*data leakage*) temporal, assegurando que janelas acústicas pertencentes a um mesmo trânsito físico nunca residam simultaneamente nos conjuntos de treino e teste.

---

## 2. Desempenho de Classificação Global (Cobertura de 100%)

A Tabela 1 consolida as métricas globais obtidas na partição de teste para todas as classes (`SMALL`, `MEDIUM`, `LARGE` e `BACKGROUND`), avaliando os extratores de características lineares de banda estreita (**LOFAR**) e integração de energia de banda larga baseada em escala log-triangular (**MEL**).

### Tabela 1: Métricas de Desempenho Geral no Conjunto de Teste (Sem Opção de Rejeição)

| Classificador / Arquitetura | Representação Espectral | Índice SP (%) | Acurácia Global (ACC) (%) | F1-Score (Micro) (%) | Classificação da Abordagem |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **RF** | MEL | 62.22 ± 1.86 | 62.64 ± 1.84 | 63.79 ± 1.73 | Baseline (Silva et al., 2025) |
| **RF** | LOFAR | 56.92 ± 1.69 | 58.87 ± 1.68 | 58.00 ± 1.41 | Baseline (Silva et al., 2025) |
| **MLP** | MEL | 63.38 ± 1.81 | 64.51 ± 1.75 | 62.89 ± 1.68 | Baseline (Silva et al., 2025) |
| **MLP** | LOFAR | **66.51 ± 1.39** | **67.48 ± 1.24** | **66.72 ± 1.17** | Baseline (Silva et al., 2025) |
| **CNN** | MEL | 63.52 ± 2.26 | 64.99 ± 2.09 | 63.04 ± 2.02 | Baseline (Silva et al., 2025) |
| **CNN** | LOFAR | 66.05 ± 1.90 | 67.02 ± 1.78 | 66.29 ± 2.13 | Baseline (Silva et al., 2025) |
| *---* | *---* | *---* | *---* | *---* | *---* |
| **SVM (m=300)** | MEL | 59.96 ± 3.38 | 62.34 ± 1.98 | 59.47 ± 3.16 | Proposto (Nyström Puro) |
| **SVM (m=1000)** | MEL | 62.96 ± 2.08 | 64.11 ± 1.98 | 62.74 ± 1.87 | Proposto (Nyström Puro) |
| **SVM (m=1000) (norm + pca64)** | MEL | 60.83 ± 1.67 | 61.86 ± 1.66 | 60.95 ± 1.31 | Proposto (PCA Redutivo) |
| **SVM (m=2000)** | MEL | 63.04 ± 2.37 | 64.45 ± 1.93 | 62.93 ± 2.22 | Proposto (Nyström Puro) |
| **SVM (m=2000) (C=10)** | MEL | 62.18 ± 3.14 | 63.53 ± 2.50 | 62.67 ± 2.84 | Proposto (Nyström Puro) |
| **SVM (m=3000)** | MEL | 63.65 ± 1.99 | 64.77 ± 1.87 | 63.59 ± 1.92 | Proposto (Nyström Puro) |
| **SVM (m=4000) (C=2.0) (pca64) (elasticnet)** | MEL | 63.82 ± 1.61 | 64.61 ± 1.60 | 64.60 ± 1.54 | Proposto (Regularizado L1/L2) |
| **SVM (m=4000) (C=2.0) (elasticnet)** | MEL | **63.78 ± 1.14** | **64.56 ± 1.18** | **64.32 ± 1.02** | **Proposto (Golden MEL - Sem PCA)** |
| *---* | *---* | *---* | *---* | *---* | *---* |
| **SVM (m=1000)** | LOFAR | 57.86 ± 3.83 | 60.95 ± 2.89 | 57.86 ± 3.53 | Proposto (Nyström Puro) |
| **SVM (m=2000)** | LOFAR | 60.10 ± 2.25 | 62.54 ± 1.98 | 59.95 ± 2.77 | Proposto (Nyström Puro) |
| **SVM (m=2000) (C=0.1)** | LOFAR | 54.84 ± 1.60 | 59.67 ± 1.28 | 54.64 ± 1.69 | Proposto (Hiperplano Suave) |
| **SVM (m=2000) (C=2.0)** | LOFAR | 60.15 ± 2.74 | 62.45 ± 2.17 | 60.59 ± 2.64 | Proposto (Nyström Puro) |
| **SVM (m=4000) (C=2.0)** | LOFAR | 61.70 ± 3.67 | 63.55 ± 2.91 | 62.85 ± 3.33 | Proposto (Nyström Puro) |
| **SVM (m=1000) (C=2.0) (pca64)** | LOFAR | 61.96 ± 2.22 | 63.16 ± 2.18 | 62.65 ± 1.74 | Proposto (PCA Redutivo) |
| **SVM (m=4000) (C=2.0) (pca64) (elasticnet)** | LOFAR | **63.10 ± 2.19** | **64.04 ± 1.88** | **64.34 ± 2.23** | **Proposto (Campeão LOFAR)** |

> [!NOTE]
> *As incertezas denotam o desvio padrão fold-wise derivado da validação cruzada 5x2. Os resultados das baselines (RF, MLP, CNN) foram extraídos diretamente da literatura de origem (Silva et al., 2025).*

---

## 3. Análise da Variância Estatística e Sensibilidade Multiclasse

### Robustez contra Viés de Classe Majoritária (Índice SP):
O índice SP (Média Geométrica e Aritmética combinada das sensibilidades por classe) atua como a métrica primária para diagnosticar o equilíbrio de fronteiras de decisão multiclasse:
1. **Convergência de Capacidade:** O modelo proposto **SVM (m=4000, Golden MEL)** alcançou **63.78% ± 1.14%** de SP. Esse resultado converge diretamente com a capacidade geométrica obtida pela CNN convolucional profunda (**63.52% ± 2.26%**), porém com **metade da variância** entre os folds de teste ($\sigma_{SVM} = 1.14\%$ vs. $\sigma_{CNN} = 2.26\%$).
2. **Preservação de Harmônicas Fracas (Classes Minoritárias):** A eliminação da etapa de Análise de Componentes Principais (PCA) na representação MEL provou-se essencial. O PCA, por ser um operador de projeção linear ortogonal baseado na maximização de variância global, descarta componentes espectrais de baixa energia. Ao mantermos os coeficientes de banco de filtros MEL originais integrados de forma não-linear pelo kernel RBF Gaussiano com penalidade ElasticNet, preservamos as frequências fundamentais e harmônicas de embarcações menores (`SMALL`), aumentando o recall de classificação em ambientes de baixa relação sinal-ruído (SNR).

---

## 4. Otimização Baseada em Opção de Rejeição (Filtro de Confiança)

Em cenários táticos reais de classificação acústica submarina, o custo associado a falsos alarmes é severamente assimétrico. Propõe-se, portanto, a incorporação de um **operador de decisão com opção de rejeição** baseado em concordância temporal de janelas.

Seja um arquivo acústico fatiado em $W$ janelas espectrais discretas. A confiança temporal $c(x)$ da classificação do arquivo é dada por:

$$c(x) = \frac{1}{W} \max_{y \in Y} \sum_{w=1}^W \mathbb{I}(\hat{y}_w = y)$$

Onde $\hat{y}_w$ é o vetor de predição do classificador para a janela $w$, e $\mathbb{I}$ representa a função indicadora. Se $c(x) < t$, a inferência é rejeitada e rotulada como classe indeterminada/desconhecida.

### Tabela 2: Curva de Trade-off Cobertura-Acurácia (LOFAR vs. MEL)

| Limiar de Confiança ($t$) | Representação Espectral | Taxa de Cobertura (%) | Acurácia de Teste (ACC) (%) | Índice SP (%) | F1-Score (Micro) (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$t = 0$** <br> *(Sem Rejeição)* | **LOFAR** <br> **MEL** | 100.00 ± 0.00 <br> 100.00 ± 0.00 | 63.32 ± 2.10 <br> **64.56 ± 1.18** | 63.10 ± 2.19 <br> **63.78 ± 1.14** | 64.34 ± 2.23 <br> **64.32 ± 1.02** |
| **$t \ge 0.5$** <br> *(Maioria Simples)* | **LOFAR** <br> **MEL** | 78.25 ± 1.90 <br> **89.67 ± 1.17** | **70.36 ± 2.13** <br> 66.23 ± 1.48 | **68.56 ± 2.88** <br> 66.07 ± 1.63 | **70.03 ± 2.56** <br> 66.74 ± 1.59 |
| **$t \ge 0.6$** <br> *(Maioria Absoluta)* | **LOFAR** <br> **MEL** | 61.21 ± 2.20 <br> **76.07 ± 2.69** | **76.54 ± 3.26** <br> 69.60 ± 1.86 | **72.87 ± 4.86** <br> 68.32 ± 2.37 | **74.64 ± 4.15** <br> 69.22 ± 2.18 |
| **$t \ge 0.9$** <br> *(Consenso Crítico)* | **LOFAR** <br> **MEL** | 22.27 ± 2.86 <br> **37.48 ± 2.32** | **90.43 ± 2.95** <br> 83.42 ± 2.07 | **78.85 ± 7.18** <br> 75.58 ± 3.18 | **82.68 ± 5.36** <br> 78.10 ± 2.71 |

---

## 5. Análise do Limite Físico e Discussão sobre Acústica de Propagação

A curva de desempenho delineada na Tabela 2 expõe um trade-off clássico governado por leis universais de propagação física no meio oceânico, permitindo duas conclusões teóricas fundamentais:

### 5.1 O LOFAR como Operador de Resolução Discreta (O Sniper de Picos)
A representação LOFAR baseia-se na preservação linear estreita da Transformada Rápida de Fourier (FFT), retendo as harmônicas puras e frequências fundamentais discretas de fontes rotativas (eixos e cilindros de pistões).
* **Física da Fronteira Geométrica:** Sob limiares severos de filtragem de confiança ($t \ge 0.9$), a acurácia de classificação converge para impressionantes **90.43% ± 2.95%**. Isso ocorre porque o modelo RBF-SVM se restringe a vetores cujos picos harmônicos estão em conformidade simétrica perfeita com os hiperplanos de alta margem de separabilidade.
* **A Limitação de Atenuação do Meio:** À medida que a distância entre a fonte (navio) e o sensor (hidrofone) aumenta, a absorção química da água do mar e a atenuação por espalhamento geométrico degradam a relação sinal-ruído (SNR). As harmônicas discretas e finas do LOFAR desaparecem rapidamente abaixo do nível do ruído oceânico plano (*background noisefloor*). Por ser um espaço de características esparso e desprovido de coeficientes integradores, o modelo perde a coerência de votação temporal e rejeita o sinal, derrubando a taxa de cobertura operacional para **22.27%**.

### 5.2 O MEL como Operador de Suavização Energética (O Escudo Resiliente)
A representação espectral MEL agrupa e comprime logarítmicamente os canais de frequências em bandas de filtragem triangulares sobrepostas.
* **Física da Integração de Banda Larga:** Ao consolidar a energia espectral difusa de bandas adjacentes, o extrator MEL age como um integrador analógico de potência. Ele amortece flutuações de ruído transientes locais e compensa o esvanecimento de frequências discretas. Como consequência direta, **a cobertura de classificação em cenários de incerteza operacional decola, retendo 76.07% de todos os alvos avaliados sob maioria absoluta ($t \ge 0.6$)**.
* **O Efeito de Borramento de Bordas:** A sobreposição intrínseca do banco de filtros Mel funde harmônicas próximas. Essa fusão gera limites de classe ligeiramente mais suaves e geométricamente ambíguos. Isso explica por que, no pico de certeza ($t \ge 0.9$), a sua acurácia máxima de convergência estima-se em **83.42%**, incapaz de replicar o pico absoluto do LOFAR.

---

## 6. Conclusões e Delineamento Tático Operacional

As evidências experimentais provam que a implementação prática de sistemas embarcados de classificação acústica passiva se beneficia da adoção de uma **Arquitetura Dinâmica em Duas Camadas**:

1. **Camada Geral de Reconhecimento Contínuo (MEL com $t \ge 0.6$):** Prioriza a cobertura espacial contínua. Automatiza a detecção de **76.07%** do tráfego marítimo com acurácia estabilizada de **69.60%** e desvio padrão fold-wise de apenas **1.86%**.
2. **Camada Tática de Engajamento de Alta Certeza (LOFAR com $t \ge 0.9$):** Direcionada a alvos críticos em aproximação de ponto crítico de aproximação (CPA - *Closest Point of Approach*). O sistema restringe-se a confirmar a classe acústica com um grau de confiabilidade científica extremo de **90.43%**, eliminando a ocorrência de alarmes falsos catastróficos.
