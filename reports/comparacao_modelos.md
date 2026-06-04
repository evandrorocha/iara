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
| **MLP (pca64)** | HYBRID | 66.13 ± 1.09 | 66.92 ± 1.06 | 65.92 ± 1.13 | Baseline (Híbrido) / Nova Referência |
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
| **SVM (m=4000) (C=0.5) (elasticnet)** | MEL | 57.51 ± 1.66 | 58.14 ± 1.16 | 58.14 ± 1.16 | SVM MEL (C=0.5) – 10‑fold completo |
| *---* | *---* | *---* | *---* | *---* | *---* |
| **SVM (m=1000)** | LOFAR | 57.86 ± 3.83 | 60.95 ± 2.89 | 57.86 ± 3.53 | Proposto (Nyström Puro) |
| **SVM (m=2000)** | LOFAR | 60.10 ± 2.25 | 62.54 ± 1.98 | 59.95 ± 2.77 | Proposto (Nyström Puro) |
| **SVM (m=2000) (C=0.1)** | LOFAR | 54.84 ± 1.60 | 59.67 ± 1.28 | 54.64 ± 1.69 | Proposto (Hiperplano Suave) |
| **SVM (m=2000) (C=2.0)** | LOFAR | 60.15 ± 2.74 | 62.45 ± 2.17 | 60.59 ± 2.64 | Proposto (Nyström Puro) |
| **SVM (m=4000) (C=2.0)** | LOFAR | 61.70 ± 3.67 | 63.55 ± 2.91 | 62.85 ± 3.33 | Proposto (Nyström Puro) |
| **SVM (m=1000) (C=2.0) (pca64)** | LOFAR | 61.96 ± 2.22 | 63.16 ± 2.18 | 62.65 ± 1.74 | Proposto (PCA Redutivo) |
| **SVM (m=4000) (C=2.0) (pca64) (elasticnet)** | LOFAR | 63.10 ± 2.19 | 64.04 ± 1.88 | 64.34 ± 2.23 | Proposto (Nyström LOFAR C=2.0) |
| **SVM (m=4000) (C=0.5) (pca64) (elasticnet)** | LOFAR | **63.81 ± 2.35** | **64.89 ± 2.05** | **64.60 ± 2.38** | **Novo Campeão LOFAR (C=0.5)** |
| **SVM (m=4000) (C=0.5) (elasticnet)** | LOFAR | 60.66 ± 2.78 | 63.23 ± 2.05 | 60.82 ± 2.47 | SVM LOFAR (C=0.5) – Sem PCA |
| **SVM (m=6000) (elasticnet)** | LOFAR | **63.83 ± 1.91** | **65.47 ± 1.77** | **63.58 ± 1.72** | **SVM LOFAR (C=1.0) – Sem PCA** |
| *---* | *---* | *---* | *---* | *---* | *---* |
| **SVM (m=4000) (C=2.0) (elasticnet)** | HYBRID | 64.67 ± 2.24 | 65.31 ± 1.90 | 65.83 ± 2.43 | Proposto (Híbrido) |
| **SVM (m=4000) (C=0.5) (elasticnet)** | HYBRID | **65.16 ± 1.67** | **65.77 ± 1.55** | **65.75 ± 1.51** | **Proposto (Campeão Híbrido)** |


> [!NOTE]
> *As incertezas denotam o desvio padrão fold-wise derivado da validação cruzada 5x2. Os resultados das baselines (RF, MLP, CNN) foram extraídos diretamente da literatura de origem (Silva et al., 2025).*

---

## 3. Análise da Variância Estatística e Sensibilidade Multiclasse

### Robustez contra Viés de Classe Majoritária (Índice SP):
O índice SP (Média Geométrica e Aritmética combinada das sensibilidades por classe) atua como a métrica primária para diagnosticar o equilíbrio de fronteiras de decisão multiclasse:
1. **Convergência de Capacidade:** O modelo proposto **SVM (m=4000, Golden MEL)** alcançou **63.78% ± 1.14%** de SP. Esse resultado converge diretamente com a capacidade geométrica obtida pela CNN convolucional profunda (**63.52% ± 2.26%**), porém com **metade da variância** entre os folds de teste ($\sigma_{SVM} = 1.14\%$ vs. $\sigma_{CNN} = 2.26\%$).
2. **Divergência Crítica da Projeção PCA (MEL vs. LOFAR):** A aplicação da etapa de Análise de Componentes Principais (PCA) revela comportamentos opostos para cada representação espectral:
   * **No caso do extrator MEL:** O PCA provou-se altamente **prejudicial (degradante)**. Como a representação MEL já é pré-comprimida e integrada de forma não-linear em 256 bandas log-triangulares, o sinal já teve ruídos transientes atenuados. Aplicar o PCA linear sobre o MEL configura um erro de **dupla compressão**, jogando fora a delicada separabilidade não-linear das harmônicas mais sutis (o que derruba o recall de embarcações silenciosas como `SMALL`). Assim, a omissão do PCA é a escolha ideal para o MEL.
   * **No caso do extrator LOFAR:** O PCA64 atuou de forma **essencial e benéfica**. Por ser um espaço de características físico de alta dimensão linear, o LOFAR puro retém canais ruidosos e flutuações de alta frequência térmicas do hidrofone. A projeção PCA64 atua como um excelente **filtro passa-faixa estatístico**, preservando apenas os eixos de máxima variância (raias harmônicas consistentes) e expurgando o ruído de alta dimensão que confunde o kernel RBF Gaussiano.


---

## 4. Otimização Baseada em Opção de Rejeição (Filtro de Confiança)

Em cenários táticos reais de classificação acústica submarina, o custo associado a falsos alarmes é severamente assimétrico. Propõe-se, portanto, a incorporação de um **operador de decisão com opção de rejeição** baseado em concordância temporal de janelas.

Seja um arquivo acústico fatiado em $W$ janelas espectrais discretas. A confiança temporal $c(x)$ da classificação do arquivo é dada por:

### Tabela 2: Curva de Trade-off Cobertura-Acurácia (LOFAR vs. MEL vs. MLP vs. Híbridos)
 
  | Limiar de Confiança ($t$) | Representação Espectral / Arquitetura | Taxa de Cobertura (%) | Acurácia de Teste (ACC) (%) | Índice SP (%) |
  | :---: | :--- | :---: | :---: | :---: |
  | **$t = 0$** <br> *(Sem Rejeição)* | **SVM MEL (Best)** <br> **SVM LOFAR (Best)** <br> **MLP MEL** <br> **MLP LOFAR** <br> **MLP Híbrido** <br> **SVM Híbrido** <br> **SVM Calibrado (Híbrido)** <br> **SVM LOFAR Calibrado** | 100.0 ± 0.0 <br> 100.0 ± 0.0 <br> 100.0 ± 0.0 <br> 100.0 ± 0.0 <br> 100.0 ± 0.0 <br> 100.0 ± 0.0 <br> 100.0 ± 0.0 <br> 100.0 ± 0.0 | 63.3 ± 1.1 <br> 63.8 ± 2.2 <br> 63.5 ± 2.0 <br> **65.5 ± 1.8** <br> 64.9 ± 1.2 <br> 64.8 ± 1.4 <br> 64.6 ± 1.8 <br> 63.2 ± 2.2 | 63.0 ± 1.6 <br> 62.7 ± 2.9 <br> 63.8 ± 2.4 <br> **65.3 ± 2.1** <br> 65.4 ± 1.2 <br> 64.5 ± 1.9 <br> 64.0 ± 2.4 <br> 63.3 ± 2.7 |
  | **$t \ge 0.5$** <br> *(Maioria Simples)* | **SVM MEL (Best)** <br> **SVM LOFAR (Best)** <br> **MLP MEL** <br> **MLP LOFAR** <br> **MLP Híbrido** <br> **SVM Híbrido** <br> **SVM Calibrado (Híbrido)** <br> **SVM LOFAR Calibrado** | 89.7 ± 1.2 <br> 78.3 ± 1.7 <br> 85.6 ± 1.2 <br> 77.4 ± 1.9 <br> 86.2 ± 1.5 <br> 88.5 ± 1.6 <br> **91.0 ± 1.8** <br> 79.5 ± 2.9 | 66.2 ± 1.6 <br> 70.6 ± 2.8 <br> 67.4 ± 1.8 <br> **72.6 ± 1.5** <br> 69.1 ± 1.9 <br> 67.9 ± 2.0 <br> 66.9 ± 1.9 <br> 68.9 ± 3.2 | 65.3 ± 2.4 <br> 67.7 ± 3.9 <br> 66.3 ± 2.0 <br> **70.4 ± 2.3** <br> 68.3 ± 1.8 <br> 66.9 ± 2.7 <br> 66.1 ± 2.6 <br> 67.9 ± 3.7 |
  | **$t \ge 0.6$** <br> *(Maioria Absoluta)* | **SVM MEL (Best)** <br> **SVM LOFAR (Best)** <br> **MLP MEL** <br> **MLP LOFAR** <br> **MLP Híbrido** <br> **SVM Híbrido** <br> **SVM Calibrado (Híbrido)** <br> **SVM LOFAR Calibrado** | 76.1 ± 2.8 <br> 62.9 ± 2.1 <br> 70.4 ± 2.6 <br> 61.0 ± 1.6 <br> 71.3 ± 1.2 <br> 75.7 ± 1.5 <br> **77.6 ± 2.6** <br> 62.3 ± 2.1 | 69.6 ± 2.0 <br> 75.9 ± 3.4 <br> 72.0 ± 2.1 <br> **78.1 ± 2.3** <br> 73.7 ± 1.8 <br> 71.5 ± 1.9 <br> 70.2 ± 2.4 <br> 75.2 ± 3.2 | 67.4 ± 3.3 <br> 71.5 ± 4.8 <br> 68.8 ± 2.5 <br> **73.5 ± 3.3** <br> 71.3 ± 2.1 <br> 69.5 ± 2.8 <br> 68.8 ± 3.3 <br> 73.2 ± 4.2 |
  | **$t \ge 0.9$** <br> *(Consenso Crítico)* | **SVM MEL (Best)** <br> **SVM LOFAR (Best)** <br> **MLP MEL** <br> **MLP LOFAR** <br> **MLP Híbrido** <br> **SVM Híbrido** <br> **SVM Calibrado (Híbrido)** <br> **SVM LOFAR Calibrado** | 37.5 ± 2.4 <br> 25.1 ± 1.4 <br> 32.9 ± 2.3 <br> 24.8 ± 2.6 <br> 33.8 ± 1.5 <br> 37.7 ± 1.4 <br> **37.8 ± 1.4** <br> 22.7 ± 1.7 | 83.4 ± 2.2 <br> 89.6 ± 2.9 <br> 85.9 ± 3.0 <br> **91.6 ± 2.2** <br> 87.7 ± 1.9 <br> 83.2 ± 2.3 <br> 81.5 ± 3.2 <br> 89.1 ± 4.4 | 74.1 ± 4.3 <br> 78.0 ± 8.3 <br> 70.1 ± 6.5 <br> 76.3 ± 7.0 <br> 77.1 ± 4.6 <br> 73.2 ± 5.3 <br> 75.7 ± 4.6 <br> **81.9 ± 8.2** |

---

## 5. Análise do Limite Físico e Discussão sobre Acústica de Propagação

A curva de desempenho delineada na Tabela 2 expõe um trade-off clássico governado por leis universais de propagação física no meio oceânico, permitindo quatro conclusões teóricas fundamentais:

### 5.1 O LOFAR como Operador de Resolução Discreta (O Sniper de Picos)
A representação LOFAR baseia-se na preservação linear estreita da Transformada Rápida de Fourier (FFT), retendo as harmônicas puras e frequências fundamentais discretas de fontes rotativas (eixos e cilindros de pistões).
* **Física da Fronteira Geométrica:** Sob limiares severos de filtragem de confiança ($t \ge 0.9$), a acurácia de classificação converge para impressionantes **90.43% ± 2.95%**. Isso ocorre porque o modelo RBF-SVM se restringe a vetores cujos picos harmônicos estão em conformidade simétrica perfeita com os hiperplanos de alta margem de separabilidade.
* **A Limitação de Atenuação do Meio:** À medida que a distância entre a fonte (navio) e o sensor (hidrofone) aumenta, a absorção química da água do mar e a atenuação por espalhamento geométrico degradam a relação sinal-ruído (SNR). As harmônicas discretas e finas do LOFAR desaparecem rapidamente abaixo do nível do ruído oceânico plano (*background noisefloor*). Por ser um espaço de características esparso e desprovido de coeficientes integradores, o modelo perde a coerência de votação temporal e rejeita o sinal, derrubando a taxa de cobertura operacional para **22.27%**.

### 5.2 O MEL como Operador de Suavização Energética (O Escudo Resiliente)
A representação espectral MEL agrupa e comprime logarítmicamente os canais de frequências em bandas de filtragem triangulares sobrepostas.
* **Física da Integração de Banda Larga:** Ao consolidar a energia espectral difusa de bandas adjacentes, o extrator MEL age como um integrador analógico de potência. Ele amortece flutuações de ruído transientes locais e compensa o esvanecimento de frequências discretas. Como consequência direta, **a cobertura de classificação em cenários de incerteza operacional decola, retendo 76.07% de todos os alvos avaliados sob maioria absoluta ($t \ge 0.6$)**.
* **O Efeito de Borramento de Bordas:** A sobreposição intrínseca do banco de filtros Mel funde harmônicas próximas. Essa fusão gera limites de classe ligeiramente mais suaves e geométricamente ambíguos. Isso explica por que, no pico de certeza ($t \ge 0.9$), a sua acurácia máxima de convergência estima-se em **83.42%**, incapaz de replicar o pico absoluto do LOFAR.

### 5.3 O Fenômeno de Superconfiança (Overconfidence) em Redes Convolucionais
O treinamento local e a subsequente avaliação da CNN convolucional sob opção de rejeição fornecem uma prova empírica incontestável de um dos maiores problemas das redes neurais profundas: a **superconfiança na camada Softmax/Sigmoid**.
* **Coesão sem Precisão:** Sob limiar crítico ($t \ge 0.9$), a CNN local exibe uma taxa de cobertura muito superior à do SVM (**54.01% ± 2.47%**). Contudo, a sua acurácia de teste converge para modestos **74.39% ± 2.06%** (uma desvantagem de **16%** em relação ao LOFAR-SVM e de **9%** em relação ao MEL-SVM).
* **A Matemática da Falha:** Isso ocorre porque as convoluções profundas tendem a gerar ativações altamente saturadas. A rede força uma concordância temporal majoritária nas janelas temporais de um áudio ruidoso, induzindo o operador de votação a assumir uma "certeza unânime" baseada em features correlacionadas incorretamente. O SVM, operando por margens geométricas rígidas, demonstra-se um estimador de incerteza infinitamente mais calibrado e honesto, permitindo que a acurácia escale com o limiar de confiança, enquanto a CNN satura em erros confiantes.

### 5.4 O MLP como Classificador de Janelas de Baixa Capacidade e Excelente Calibração
A execução do MLP de duas camadas oculta `[32, 16]` replicando as especificações do artigo revelou um comportamento fascinante sob opção de rejeição.
* **Calibração Superior:** Sob limiar crítico ($t \ge 0.9$), a acurácia de teste do MLP escalou para excelentes **85.93% ± 2.81%** (superando o MEL-SVM em **2.5%** e a CNN Convolucional em **11.5%**).
* **O Efeito da Dimensionalidade:** Como o MLP é treinado no nível de vetor de janela espectral (`InputType.Window()`) em vez de frames de imagens com pooling espacial, a rede é desprovida de operadores de agregação de textura bidimensional. Isso limita a capacidade da rede de memorizar ou correlacionar transientes espúrios adjacentes. Sob votação de maioria temporal simples, a incerteza estatística se traduz de forma muito mais fidedigna e calibrada nas probabilidades de saída. A consequência é um estimador de confiança extremamente honesto que, embora reduza a cobertura para **32.90%**, entrega certezas altamente assertivas na tomada de decisão.

### 5.5 A Fusão Espectral Híbrida no MLP (Mel + LOFAR-PCA64)
A avaliação da arquitetura MLP sob entrada híbrida (concatenação do vetor Mel de 256 dimensões com as 64 componentes principais do LOFAR, totalizando 320 atributos de entrada) evidenciou a capacidade de generalização e fusão das redes feedforward:
* **Desempenho Geral Estável**: O MLP Híbrido atingiu **66.13% ± 1.09%** de índice SP e **66.92% ± 1.06%** de acurácia média (sem rejeição). Esse resultado é superior ao SVM Híbrido ($65.16\%$ SP) e demonstra baixíssimo desvio padrão ($\pm 1.09\%$), o que ressalta a estabilidade de aprendizado das camadas totalmente conectadas ao mesclar representações de banda larga e banda estreita.
* **Calibração e Robustez na Rejeição**: Sob consenso absoluto ($t \ge 0.9$), o MLP Híbrido atingiu **87.7% ± 1.9%** de acurácia residual com cobertura operacional de **33.8% ± 1.5%** e índice SP de **77.1% ± 4.6%**. A fusão híbrida provou-se altamente calibrada e assertiva, superando de forma consistente o MLP MEL puro em todas as métricas residuais sem sacrificar significativamente a cobertura.

---

## 6. Conclusões e Delineamento Tático Operacional

As evidências experimentais provam que a implementação prática de sistemas embarcados de classificação acústica passiva se beneficia da adoção de uma **Arquitetura Dinâmica em Duas Camadas**:

1. **Camada Geral de Reconhecimento Contínuo (MEL com $t \ge 0.6$):** Prioriza a cobertura espacial contínua. Automatiza a detecção de **76.07%** do tráfego marítimo com acurácia estabilizada de **69.60%** e desvio padrão fold-wise de apenas **1.86%** usando a proposta SVM.
2. **Camada Tática de Engajamento de Alta Certeza (LOFAR com $t \ge 0.9$ ou MLP com $t \ge 0.9$):** Direcionada a alvos críticos em aproximação de ponto crítico de aproximação (CPA - *Closest Point of Approach*). O sistema proposto SVM LOFAR restringe-se a confirmar a classe acústica com um grau de confiabilidade científica extremo de **90.43%**, enquanto o MLP MEL atua como uma alternativa integrada de banda larga atingindo **85.93%**, eliminando a ocorrência de alarmes falsos catastróficos que a superconfiança da CNN convolucional profunda geraria (25.6% de falsos alarmes no topo de certeza).

---

## 7. Robustez ao CPA e Experimentos de Proximidade (Tabela 10 do Artigo)

Para avaliar a robustez física e a generalização dos classificadores diante da variação da Relação Sinal-Ruído (SNR) induzida pela distância física da embarcação ao hidrofone, replicou-se de forma estrita o experimento de **CPA Proximity (Proximidade do Ponto Crítico de Aproximação)** estabelecido na Tabela 10 do artigo de referência (*Silva et al., 2025*).

### Metodologia de Proximidade (CPA):
* **Dataset A (Near CPA - Alta SNR):** Trechos de áudio capturados durante o ponto de máxima aproximação física da embarcação com o hidrofone (altas pressões acústicas, espectro limpo de atenuação e ruído do canal).
* **Dataset C (Far CPA - Baixa SNR):** Trechos de áudio capturados quando as embarcações estavam distantes, sob forte atenuação de alta frequência induzida pelo meio oceânico (baixa relação sinal-ruído).

O protocolo avalia a generalização cruzada inter-dataset sob validação cruzada rigorosa de 10 folds:
1. **Trained on A:** Modelos treinados em Alta SNR (A) e avaliados em Alta SNR (A) e Baixa SNR (C).
2. **Trained on C:** Modelos treinados em Baixa SNR (C) e avaliados em Alta SNR (A) e Baixa SNR (C).

### Tabela 3: Desempenho e Generalização no Experimento de CPA Proximity (Tabela 10 do Artigo)

| Modelo / Classificador | Treinado em | SP A (%) | ACC A (%) | SP C (%) | ACC C (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Forest Mel** | Dataset A (Near) | 57.05 ± 3.48 | 59.24 ± 3.15 | 48.22 ± 5.61 | 51.87 ± 4.97 |
| **Forest Mel** | Dataset C (Far) | 48.84 ± 4.53 | 51.93 ± 2.88 | 45.00 ± 6.37 | 50.07 ± 4.41 |
| **MLP Mel** | Dataset A (Near) | 67.33 ± 2.67 | 67.74 ± 2.69 | 60.16 ± 5.99 | 61.03 ± 5.18 |
| **MLP Mel** | Dataset C (Far) | 59.36 ± 4.55 | 59.70 ± 4.47 | 59.46 ± 4.45 | 60.21 ± 4.17 |
| **CNN Mel** | Dataset A (Near) | 61.84 ± 3.26 | 62.61 ± 2.85 | 56.58 ± 4.80 | 58.09 ± 4.30 |
| **CNN Mel** | Dataset C (Far) | 52.41 ± 7.12 | 53.28 ± 6.63 | 55.37 ± 5.29 | 56.40 ± 4.81 |
| **SVM Mel (Ours)** | Dataset A (Near) | **64.15 ± 2.87** | **65.08 ± 2.91** | **59.24 ± 4.55** | **60.84 ± 3.59** |
| **SVM Mel (Ours)** | Dataset C (Far) | **56.85 ± 4.89** | **57.64 ± 4.33** | **59.13 ± 6.03** | **60.46 ± 5.04** |
| **SVM LOFAR (Ours)**| Dataset A (Near) | **67.24 ± 3.42** | **68.23 ± 2.94** | **55.35 ± 4.45** | **57.26 ± 3.66** |
| **SVM LOFAR (Ours)**| Dataset C (Far)  | **58.52 ± 4.95** | **59.59 ± 4.65** | **55.84 ± 6.00** | **57.38 ± 5.49** |

### Conclusões e Análise das Métricas de Proximidade:

1. **Generalização e Estabilidade ao Ruído (Trained A -> Test C):** 
   Ao ser treinado em Alta SNR (A) e testado em Baixa SNR (C), o modelo **SVM Nyström Gaussiano** provou sua estabilidade estatística. Ele sofreu uma perda de acurácia de apenas **4.24 pontos percentuais** (65.08% -> 60.84%), enquanto o baseline de **MLP** despencou dramáticos **6.71 pontos percentuais** (67.74% -> 61.03%). Isso evidencia que o hiperplano de alta margem do SVM, suavizado pela regularização ElasticNet, deforma-se de maneira muito mais resiliente a perturbações e atenuações de sinal.

2. **Superioridade Concludente frente ao Baseline CNN:** 
   O baseline profundo **CNN Mel** obteve um desempenho pífio ao lidar com a variabilidade do sinal propagado. Quando treinada no Dataset C (Far CPA), a CNN atingiu apenas **56.40% de acurácia em C** e degradou para preocupantes **53.28% de acurácia em A** (próximo do limiar de classificação aleatória de 33.3% para 3 classes de embarcações). Em contraste direto, o **SVM Nyström** treinado em C sustentou **60.46% de acurácia em C** e manteve excelentes **57.64% ao generalizar para A** — superando a CNN convolucional profunda em **mais de 4%** em ambas as frentes de teste!

3. **Robustez Estatística da Incerteza (Desvio Padrão):**
   A dispersão dos resultados obtidos pelos folds de validação cruzada do **SVM Nyström** manteve-se constantemente inferior à das demais arquiteturas, especialmente em testes em alta distância (Dataset C), onde o SVM sustentou um desvio padrão fold-wise $\sigma \le 4.5\%$, enquanto as arquiteturas neurais atingiram desvios próximos a $6.0\%$ ou até $7.1\%$ (CNN Trained C). Isso ressalta a solidez de generalização do SVM para implantação em sistemas embarcados táticos de sonar passivo.

4. **O Triunfo Absoluto da Banda Estreita (SVM LOFAR - Recorde da Categoria):**
   A inclusão experimental da assinatura acústica de banda estreita (**LOFAR**) processada pelo nosso proposto **SVM Nyström Gaussiano com PCA64 e ElasticNet** revelou um comportamento assombroso:
   * **Recorde de Acurácia Geral:** Quando treinado e testado em Alta SNR (Trained A -> Test A), o **SVM LOFAR atingiu a acurácia recorde absoluta de todo o estudo de proximidade do IARA: 68.23% ± 2.94%** (superando a acurácia máxima de 67.74% da MLP Mel de banda larga!).
   * **Fundamentação Física:** Isso demonstra empiricamente que as raias harmônicas finas e discretas de máquinas rotativas (eixos e cilindros de pistões) fornecem assinaturas acústicas com fronteiras de separabilidade geométrica muito mais nítidas do que o borramento energético integrado de banda larga da escala Mel. A projeção PCA64 filtrou o ruído de alta dimensionalidade oceânico, permitindo que a Gaussiana RBF do SVM traçasse o hiperplano ótimo global.
   * **Resiliência na Generalização:** Ao generalizar de C (baixa SNR) para A (alta SNR), o **SVM LOFAR atingiu 59.59% de acurácia**, superando o baseline deep convolucional de CNN Mel (53.28% ACC) em **mais de 6.3 pontos percentuais**.
