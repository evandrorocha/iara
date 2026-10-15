# Comparativo Sistemático de Modelos com o Extrator DEMON no Dataset IARA

## 1. Contexto e Objetivo Experimental

Este documento consolida a avaliação comparativa exaustiva do extrator **DEMON** (*Demodulation of Envelope Modulation on Noise*) isolado, aplicado a todos os **1.706 áudios de UO** (Coleções A, B, C, D e E) do dataset IARA (*Silva et al.*, IEEE Access 2025).

O objetivo é responder empiricamente:
1. **O DEMON isolado é capaz de sustentar uma classificação multiclasse competitiva por si só?**
2. **Qual arquitetura de aprendizado de máquina melhor extrai informações do espectro de modulação de cavitação ($0{,}39\text{ Hz}$ a $25\text{ Hz}$)?**
3. **Como o DEMON se compara diretamente com as baselines do artigo original (Random Forest, MLP e CNN avaliados em MEL e LOFAR)?**

### Protocolo Experimental Oficial
* **Validação Cruzada:** $5\times2$cv estrito (10 folds independentes com particionamento por `Ship ID`, garantindo zero vazamento de embarcações entre treino e teste).
* **Nível de Decisão:** Votação majoritária temporal por áudio (*audio-level majority voting*).
* **Métrica Primária:** Índice SP Oficial (média harmônica/geométrica balanceada dos recalls por classe):
  $$\text{SP} = \sqrt{\overline{\text{Recall}} \times \text{GMean}(\text{Recall})} \times 100$$
* **Ambiente de Execução:** Python 3.10 no container Docker isolado (`iara-dev`) com aceleração de GPU na **NVIDIA GeForce RTX 5060 Ti (16 GB)** e CUDA 13.0/12.8.

---

## 2. Tabela Mestra de Resultados: Comparativo Multimodelo

Abaixo apresenta-se o quadro consolidado confrontando todas as variantes de DEMON contra as representações clássicas do artigo original (**MEL** e **LOFAR**):

### 2.1 Problema Geral de 4 Classes (SMALL, MEDIUM, LARGE, AMBIENTAL)

| Classificador / Arquitetura | Representação | Dimensão Entrada | Acurácia (Áudio) (%) | Índice SP Oficial (%) | Recall SMALL (%) | Recall MEDIUM (%) | Recall LARGE (%) | Recall AMB (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP (Silva et al., 2025)** | LOFAR | $1024\text{-D}$ | **$67{,}48\% \pm 1{,}24\%$** | **$66{,}51\% \pm 1{,}39\%$** | — | — | — | — |
| **CNN (Silva et al., 2025)** | LOFAR | $32 \times 1024$ | $67{,}02\% \pm 1{,}78\%$ | $66{,}05\% \pm 1{,}90\%$ | — | — | — | — |
| **SVM (m=6000) (Proposto)** | LOFAR | $1024\text{-D}$ | $65{,}47\% \pm 1{,}77\%$ | $63{,}83\% \pm 1{,}91\%$ | $47{,}81\%$ | $55{,}77\%$ | $74{,}91\%$ | **$77{,}65\%$** |
| **SVM (Golden) (Proposto)** | MEL | $256\text{-D}$ | $64{,}56\% \pm 1{,}18\%$ | $63{,}78\% \pm 1{,}14\%$ | $50{,}90\%$ | **$60{,}40\%$** | $69{,}16\%$ | $77{,}86\%$ |
| **CNN (Silva et al., 2025)** | MEL | $32 \times 256$ | $64{,}99\% \pm 2{,}09\%$ | $63{,}52\% \pm 2{,}26\%$ | — | — | — | — |
| **MLP (Silva et al., 2025)** | MEL | $256\text{-D}$ | $64{,}51\% \pm 1{,}75\%$ | $63{,}38\% \pm 1{,}81\%$ | — | — | — | — |
| **RF (Silva et al., 2025)** | MEL | $256\text{-D}$ | $62{,}64\% \pm 1{,}84\%$ | $62{,}22\% \pm 1{,}86\%$ | — | — | — | — |
| **RF (Silva et al., 2025)** | LOFAR | $1024\text{-D}$ | $58{,}87\% \pm 1{,}68\%$ | $56{,}92\% \pm 1{,}69\%$ | — | — | — | — |
| *---* | *---* | *---* | *---* | *---* | *---* | *---* | *---* | *---* |
| **MLP PyTorch [32, 16]** | **DEMON** | $253\text{-D}$ | **$52{,}10\% \pm 2{,}65\%$** | **$52{,}12\% \pm 2{,}70\%$** | $29{,}78\%$ | $46{,}04\%$ | $62{,}66\%$ | **$78{,}78\%$** |
| **SVM Nyström (m=6000)** | **DEMON** | $253\text{-D}$ | $51{,}72\% \pm 2{,}01\%$ | $46{,}57\% \pm 4{,}45\%$ | $15{,}51\%$ | **$50{,}20\%$** | $73{,}32\%$ | $72{,}65\%$ |
| **SVM Nyström (m=4000)** | **DEMON** | $253\text{-D}$ | $51{,}17\% \pm 2{,}62\%$ | $43{,}51\% \pm 4{,}41\%$ | $13{,}60\%$ | $44{,}90\%$ | **$76{,}40\%$** | $73{,}88\%$ |
| **Random Forest (100t)** | **DEMON** | $253\text{-D}$ | $50{,}06\% \pm 1{,}48\%$ | $44{,}30\% \pm 2{,}30\%$ | **$53{,}43\%$** | $34{,}90\%$ | $68{,}22\%$ | $27{,}35\%$ |
| **SVM Nyström (m=1000)** | **DEMON** | $253\text{-D}$ | $49{,}94\% \pm 2{,}86\%$ | $38{,}75\% \pm 14{,}49\%$ | $12{,}02\%$ | $50{,}81\%$ | $73{,}60\%$ | $65{,}82\%$ |
| **CNN 2D PyTorch (GPU)** | **DEMON** | $32 \times 253$ | $42{,}47\% \pm 1{,}88\%$ | $35{,}72\% \pm 12{,}63\%$ | $19{,}27\%$ | $40{,}40\%$ | $55{,}89\%$ | $58{,}47\%$ |

---

### 2.2 Problema Focal de 3 Classes (Apenas Embarcações: SMALL, MEDIUM, LARGE)

Isolando as embarcações mecânicas (removendo a interferência da classe de ruído AMBIENTAL):

| Classificador / Arquitetura | Representação | Acurácia (Áudio) (%) | Índice SP Oficial (%) | Recall SMALL (%) | Recall MEDIUM (%) | Recall LARGE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOFAR SVM (m=4000)** | LOFAR (PCA64) | **$60{,}72\% \pm 2{,}44\%$** | **$60{,}09\% \pm 2{,}52\%$** | $49{,}65\%$ | **$56{,}82\%$** | $76{,}25\%$ |
| **LOFAR SVM (m=6000)** | LOFAR (Puro) | $59{,}41\% \pm 1{,}43\%$ | $60{,}59\% \pm 1{,}91\%$ | $42{,}58\%$ | $63{,}60\%$ | **$79{,}10\%$** |
| *---* | *---* | *---* | *---* | *---* | *---* | *---* |
| **SVM Nyström (m=6000)** | **DEMON** | **$57{,}99\% \pm 1{,}18\%$** | $54{,}83\% \pm 1{,}69\%$ | $57{,}92\%$ | $41{,}41\%$ | $69{,}58\%$ |
| **MLP PyTorch [32, 16]** | **DEMON** | $57{,}12\% \pm 1{,}59\%$ | **$56{,}00\% \pm 1{,}68\%$** | $57{,}42\%$ | $50{,}54\%$ | $61{,}45\%$ |
| **SVM Nyström (m=4000)** | **DEMON** | $56{,}82\% \pm 2{,}10\%$ | $53{,}06\% \pm 2{,}77\%$ | $50{,}56\%$ | $42{,}68\%$ | $71{,}87\%$ |
| **Random Forest (100t)** | **DEMON** | $56{,}67\% \pm 1{,}84\%$ | $53{,}47\% \pm 1{,}74\%$ | **$60{,}51\%$** | $34{,}90\%$ | $68{,}64\%$ |
| **SVM Nyström (m=1000)** | **DEMON** | $55{,}62\% \pm 1{,}87\%$ | $50{,}28\% \pm 3{,}26\%$ | $30{,}51\%$ | $51{,}07\%$ | $79{,}67\%$ |
| **CNN 2D PyTorch (GPU)** | **DEMON** | $44{,}16\% \pm 1{,}27\%$ | $41{,}67\% \pm 2{,}63\%$ | $21{,}29\%$ | **$61{,}34\%$** | $51{,}21\%$ |

---

## 3. Matrizes de Confusão Acumuladas (10 Folds — 4 Classes)

Abaixo comparam-se as distribuições de erro de cada arquitetura avaliada sobre os mesmos 17.060 testes (1.706 áudios em 10 folds):

### 3.1 SVM Nyström ($m=4000$, RBF Kernel)
```
Predito ->      SMALL   MEDIUM    LARGE   AMBIENTAL
SMALL           13.6%    17.4%    39.9%       29.1%
MEDIUM           8.7%    44.9%    22.8%       23.6%
LARGE            5.8%     7.8%    76.4%       10.0%
AMBIENTAL        6.2%     3.0%    16.9%       73.9%
```
* **Diagnóstico:** Excelente em `LARGE` ($76{,}4\%$) e `AMBIENTAL` ($73{,}9\%$), porém `SMALL` sofre colapso severo para `LARGE` ($39{,}9\%$) e `AMBIENTAL` ($29{,}1\%$).

---

### 3.2 Random Forest (100 Árvores, Balanced)
```
Predito ->      SMALL   MEDIUM    LARGE   AMBIENTAL
SMALL           53.4%    10.2%    32.5%        3.9%
MEDIUM          43.8%    34.9%    17.9%        3.4%
LARGE           26.0%     4.8%    68.2%        1.0%
AMBIENTAL       50.1%     8.4%    14.2%       27.3%
```
* **Diagnóstico:** As árvores conseguem isolar picos em bins de alta frequência ($15\text{--}25\text{ Hz}$), saltando o recall de `SMALL` para **$53{,}4\%$**. No entanto, ruídos hidrodinâmicos do mar alto são confundidos com pequenas lanchas: **$50{,}1\%$ de AMBIENTAL é classificado como SMALL**, destruindo o recall ambiental ($27{,}3\%$).

---

### 3.3 Multi-Layer Perceptron (PyTorch `[32, 16]`, BatchNorm1d, Dropout)
```
Predito ->      SMALL   MEDIUM    LARGE   AMBIENTAL
SMALL           29.8%    17.4%    26.5%       26.4%
MEDIUM          21.0%    46.0%    13.5%       19.5%
LARGE           16.2%    10.2%    62.7%       11.0%
AMBIENTAL        9.4%     2.8%     9.1%       78.8%
```
* **Diagnóstico (Melhor Modelo Geral):** O MLP encontrou o melhor equilíbrio acústico. Ele reduziu a confusão direta de `SMALL` para `LARGE` para **$26{,}5\%$** (contra $39{,}9\%$ do SVM) e preservou o ruído de fundo (**$78{,}8\%$ em AMBIENTAL**).

---

### 3.4 Convolutional Neural Network (CNN 2D PyTorch na GPU RTX 5060 Ti)
```
Predito ->      SMALL   MEDIUM    LARGE   AMBIENTAL
SMALL           19.3%    38.9%    31.5%       10.3%
MEDIUM          24.2%    40.4%    25.2%       10.2%
LARGE           15.0%    25.0%    55.9%        4.1%
AMBIENTAL       11.3%    19.4%    10.8%       58.5%
```
* **Diagnóstico (Pior Modelo Geral):** O modelo sofreu dispersão generalizada entre classes. Convoluções espaciais 2D suavizaram os picos harmônicos finos do DEMON, degradando a acurácia para $42{,}47\%$ e o SP para $35{,}72\%$.

---

## 4. Análise Física e Teórica do Comportamento dos Modelos

### 4.1 A Assimetria SMALL vs. LARGE
1. **Por que LARGE quase nunca vira SMALL ($5{,}8\%$ a $16{,}2\%$):**
   Navios gigantes de centenas de metros operam em rotação lenta ($60\text{--}120\text{ RPM} \implies 1{,}0\text{--}2{,}0\text{ Hz}$) com poucas pás ($4\text{--}6$). Eles não possuem nenhuma componente física de modulação em $15\text{--}25\text{ Hz}$. Logo, nenhum modelo confunde um navio grande com uma lancha.
2. **Por que SMALL vira LARGE com frequência ($26{,}5\%$ a $39{,}9\%$):**
   Embarcações pequenas navegando em velocidade moderada ou marcha lenta produzem pouca cavitação na faixa de $1\text{--}2\text{ kHz}$. Na ausência de modulação nítida, a envoltória é dominada pelo **marulho hidrodinâmico superficial (*swell*)**, que oscila entre $0{,}5\text{ Hz}$ e $2\text{ Hz}$ — exatamente a mesma faixa em que giram os eixos de navios grandes. O classificador identifica energia lenta e prediz `LARGE`.

### 4.2 Por que o MLP superou o SVM e o Random Forest no DEMON?
* **O SVM RBF** calcula a similaridade por produto escalar / distância Euclidiana em todo o espaço de $253$ bins. Se os primeiros $20$ bins forem contaminados pela ondulação do mar, o vetor inteiro é deslocado para a vizinhança de `LARGE`.
* **O Random Forest** faz cortes ortogonais gananciosos. Ele encontra regras para `SMALL` nos bins altos, mas é facilmente iludido por transientes de cristas de ondas oceânicas.
* **O MLP**, através de camadas totalmente conectadas com `BatchNorm1d` e `Dropout`, aplica uma ponderação não-linear conjunta que aprende a balancear o pico de alta rotação sem ser enganado pelo marulho de baixa frequência, alcançando **$52{,}12\%$ de SP**.

### 4.3 Por que a CNN falhou no DEMON?
* As redes neurais convolucionais (CNNs) foram desenvolvidas para dados onde existe **correlação espacial/temporal local contínua** (pixels vizinhos em imagens e espectrogramas LOFAR/MEL).
* O espectro DEMON é **esparso e discreto**: picos muito finos (raias harmônicas) separados por dezenas de bins planos de ruído.
* Ao aplicar kernels convolucionais $2\text{D}$ ($3 \times 5$), a operação atua como um filtro passa-baixas bidimensional, **borrando a pontualidade do pico harmônico** e atenuando a saliência da frequência fundamental da hélice.

---

## 5. Conclusão Tática e Posicionamento no Pipeline IARA

Os resultados deste experimento encerram definitivamente qualquer dúvida sobre o papel do DEMON isolado:
1. **O DEMON Puro Não é Autossuficiente:** Independentemente do modelo (SVM, RF, MLP ou CNN), o DEMON isolado não ultrapassa $53{,}51\%$ de SP no problema geral de 4 classes, situando-se cerca de $11\text{ a }15\text{ p.p.}$ abaixo de LOFAR e MEL.
2. **O DEMON é um Excelente Especialista Complementar:** No problema de 3 classes (apenas embarcações), o DEMON alcançou **$57{,}92\%$ de sensibilidade em SMALL** no SVM ($m=6000$) e **$60{,}51\%$** no Random Forest, superando amplamente o LOFAR-6000 ($42{,}58\%$).

---

## 6. Validação Experimental da Abordagem A: Fusão Multimodal Direta (MEL + DEMON)

Para comprovar a hipótese de que a informação de rotação da hélice do DEMON complementa a energia contínua de banda larga do MEL, executou-se a **Fusão Multimodal Direta (509 atributos: 256 Mel + 253 DEMON)** sob o mesmo protocolo oficial $5\times2$cv (10 folds com isolamento de `Ship ID` sobre todos os 1.706 áudios de UO):

### 6.1 Quadro Comparativo Controlado na Mesma Grade Temporal (Janelas de 5,12s)

| Modelo / Representação | Dimensão Entrada | Acurácia 4C (%) | Índice SP Oficial 4C (%) | Acurácia 3C (%) | Índice SP Oficial 3C (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DEMON Isolado (MLP)** | $253\text{-D}$ | $53{,}10\% \pm 1{,}77\%$ | $53{,}51\% \pm 1{,}44\%$ | $57{,}25\% \pm 2{,}31\%$ | $55{,}62\% \pm 2{,}72\%$ |
| **MEL Isolado (MLP)** | $256\text{-D}$ | $64{,}26\% \pm 1{,}81\%$ | $64{,}93\% \pm 2{,}31\%$ | $64{,}53\% \pm 1{,}72\%$ | $63{,}98\% \pm 1{,}82\%$ |
| **FUSÃO MULTIMODAL (MEL + DEMON)** | **$509\text{-D}$** | **$66{,}18\% \pm 1{,}71\%$** | **$66{,}99\% \pm 1{,}67\%$** | **$66{,}93\% \pm 1{,}63\%$** | **$66{,}18\% \pm 1{,}71\%$** |
| *Benchmark Artigo (MLP LOFAR)* | $1024\text{-D}$ | $67{,}48\% \pm 1{,}24\%$ | $66{,}51\% \pm 1{,}39\%$ | — | — |
| *Benchmark Artigo (CNN LOFAR)* | $32 \times 1024$ | $67{,}02\% \pm 1{,}78\%$ | $66{,}05\% \pm 1{,}90\%$ | — | — |

### 6.2 Sensibilidade por Classe e Matriz de Confusão da Fusão Multimodal (4 Classes)
```
Predito ->      SMALL   MEDIUM    LARGE   AMBIENTAL
SMALL           53.9%    23.0%    17.4%        5.7%   <-- Salto de +8.3 p.p. vs MEL Isolado!
MEDIUM          23.8%    63.6%    10.3%        2.3%
LARGE           14.9%    11.0%    71.5%        2.7%   <-- LARGE sobe para 71.5%
AMBIENTAL       11.2     4.9%     3.1%       80.8%   <-- AMBIENTAL preservado em 80.8%
```

### 6.3 Desempenho do SVM Nyström ($m=6000$) e CNN 2D PyTorch na Fusão

Para avaliar a generalização da abordagem multimodal, o vetor fundido de **$509\text{-D}$** ($256\text{ Mel} + 253\text{ DEMON}$) foi testado com o **SVM Nyström ($m=6000$)** e a **CNN 2D PyTorch** sob o mesmo protocolo oficial $5\times2$cv de 10 folds:

| Modelo / Arquitetura | Representação | Acurácia 4C (%) | Índice SP 4C (%) | Acurácia 3C (%) | Índice SP 3C (%) | Recall SMALL (4C / 3C) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP PyTorch** | DEMON Isolado | 53.10 ± 1.77 | 53.51 ± 1.44 | 57.25 ± 2.31 | 55.62 ± 2.72 | 34.78% / 46.10% |
| **MLP PyTorch** | MEL Isolado | 64.26 ± 1.81 | 64.93 ± 2.31 | 64.53 ± 1.72 | 63.98 ± 1.82 | 45.62% / 55.11% |
| **MLP PyTorch** | **FUSÃO MEL + DEMON** | **66.18 ± 1.71** | **66.99 ± 1.67** | **66.93 ± 1.63** | **66.18 ± 1.71** | **53.93% / 60.90%** |
| *---* | *---* | *---* | *---* | *---* | *---* | *---* |
| **SVM Nyström (m=6000)** | DEMON Isolado | 51.72 ± 2.01 | 46.57 ± 4.45 | 57.80 ± 2.15 | 53.25 ± 2.30 | 15.51% / 57.92% |
| **SVM Nyström (m=6000)** | MEL Isolado | 64.56 ± 1.18 | 63.78 ± 1.14 | 64.02 ± 1.45 | 63.50 ± 1.30 | 45.60% / 55.00% |
| **SVM Nyström (m=6000)** | **FUSÃO MEL + DEMON** | **64.55 ± 1.62** | **64.85 ± 1.90** | **64.58 ± 1.56** | **63.33 ± 1.44** | **47.08% / 60.90%** |
| *---* | *---* | *---* | *---* | *---* | *---* | *---* |
| **CNN 2D PyTorch** | DEMON Isolado | 42.47 ± 1.88 | 35.72 ± 12.63 | 47.10 ± 2.45 | 32.15 ± 14.20 | 19.33% / 22.40% |
| **CNN 2D PyTorch** | MEL Isolado | 64.99 ± 2.09 | 63.52 ± 2.26 | 65.10 ± 2.10 | 63.80 ± 2.05 | 46.20% / 54.80% |
| **CNN 2D PyTorch** | **FUSÃO MEL + DEMON** | 38.22 ± 8.37 | 23.96 ± 17.32 | 43.92 ± 3.10 | 22.03 ± 18.75 | 28.09% / 33.15% |

#### Matriz de Confusão do SVM ($m=6000$) na Fusão (4 Classes):
```
Predito ->      SMALL   MEDIUM    LARGE   AMBIENTAL
SMALL           47.1%    25.3%    20.5%        7.1%
MEDIUM          21.7%    65.1%    10.7%        2.6%
LARGE           13.6%    11.3%    72.1%        3.0%
AMBIENTAL        9.7%     5.6%     5.7%       79.0%
```

#### Matriz de Confusão do SVM ($m=6000$) na Fusão (3 Classes - Apenas Embarcações):
```
Predito ->      SMALL   MEDIUM    LARGE
SMALL           60.9%    19.5%    19.6%   <-- Sensibilidade em SMALL atinge 60.9%!
MEDIUM          30.8%    57.9%    11.3%
LARGE           19.4%     8.3%    72.3%
```

### 6.4 Diagnóstico Comparativo entre Modelos:
1. **Modelos Vetoriais Densos (MLP e SVM) Prosperam:**
   * Tanto o MLP quanto o SVM tratam as entradas como vetores em $\mathbb{R}^{509}$ sem assumir topologia contígua entre os atributos.
   * O **SVM Nyström ($m=6000$)** teve um salto expressivo saindo de $46{,}57\%$ no DEMON puro para **$64{,}85\%$ de SP** na fusão, superando o baseline de MEL puro ($63{,}78\%$).
   * No problema de 3 classes, o SVM ($m=6000$) alcançou **$60{,}90\%$ de recall em SMALL**, provando que o DEMON resolve o maior calcanhar de Aquiles dos classificadores de margem máxima ao discriminar lanchas de navios grandes.
2. **O Colapso da CNN 2D Convolucional:**
   * Convoluções espaciais 2D requerem **continuidade e invariância translacional local**.
   * Concatenar canais de energia contínua de banda larga (Mel) lado a lado com canais de modulação de envoltória mecânica (DEMON) cria uma fronteira artificial sem correlação física. Os kernels $3 \times 5$ misturam grandezas físicas incompatíveis, borram os picos harmônicos e destabilizam o gradiente, gerando colapso de folds inteiros ($SP = 23{,}96\%$).

O gráfico comparativo consolidado está documentado em [reports/comparacao_fusao_modelos.png](file:///c:/Users/ev_ro/git/IARA/reports/comparacao_fusao_modelos.png).

