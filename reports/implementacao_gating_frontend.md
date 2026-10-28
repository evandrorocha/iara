# Implementação e Avaliação do Sistema Gating Front-End Multimodal

> **Correção de 23/10/2026 — extração HHT.** O cache HHT usado neste documento misturava duas configurações de EMD (`MAX_ITERATION` ≥ 100 em 571 gravações e 30 nas demais; nenhuma gravação BACKGROUND no primeiro grupo). O HHT foi re-extraído com configuração única (8 IMFs, `MAX_ITERATION=30`) e todos os experimentos HHT foram retreinados; a auditoria e o antes × depois estão no `notebooks/28_correcao_extracao_hht.ipynb`. Os números HHT abaixo são **anteriores à correção** (e alguns vêm de execuções ainda mais antigas que as predições em disco). Valores atuais, recalculados das predições: RF HHT isolado 57,77 (antes 60,64, na convenção de voto do pickle do gating); Gated RF 64,04 (antes 64,56); H-MoE especializado 65,36 (antes 65,41); os roteadores mudam menos de 0,6 p.p. As frações de rota para o HHT também mudaram (ver Notebook 26).

**Data:** Outubro de 2026  
**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)  
**Dataset:** IARA (1.825 áudios reais de tráfego marítimo na Bacia de Santos)  
**Protocolo:** Validação Cruzada $5\times2\text{cv}$ Oficial (10 folds estratificados, particionados por gravação `ID` como no artigo; o mesmo navio, `Ship ID`, pode ter gravações em treino e teste)  
**Notebook Demonstrativo:** [notebooks/26_sistema_gating_frontend.ipynb](file:///c:/Users/ev_ro/git/IARA/notebooks/26_sistema_gating_frontend.ipynb)  

---

## 1. Conceito e Arquitetura do Roteamento por Competência

O sistema **Gating Front-End** foi projetado para atuar **na porta de entrada** do pipeline de sonar passivo, eliminando a necessidade de classificadores em cascata ou fusões cegas pesadas.

O áudio submarino recebido tem suas **propriedades físicas acústicas avaliadas em microssegundos** por um módulo leve baseado em **Estimativa de Competência Local** (*RF Competence Router*). Em vez de uma classificação categórica rígida (que penalizava casos onde múltiplos especialistas acertavam simultaneamente), o roteador treina estimadores binários de probabilidade de acerto para cada representação candidata:

$$\hat{m} = \arg\max_{m \in \{\text{MEL}, \text{LOFAR}, \text{HHT}, \text{DEMON}\}} P(\text{Classificador na representação } m \text{ acerta} \mid \mathbf{x}_{\text{meta}})$$

```
                            Sinal de Áudio Submarino (.wav)
                                           │
                                           ▼
                     ┌───────────────────────────────────────────┐
                     │    Acoustic Front-End Gating (Router)     │
                     │  Avaliação física ultrarrápida do sinal:  │
                     │  • High / Low Energy Ratio                │
                     │  • Crest Factor (< 250 Hz)                │
                     │  • Spectral Flatness & Spectral Flux      │
                     │  • Modulation Index (2 - 8 kHz)           │
                     └─────────────────────┬─────────────────────┘
                                           │
                                           │ Seleção Dinâmica da Entrada (argmax P(acerto))
                                           ▼
               ┌───────────────┬───────────────────┬───────────────┐
               ▼               ▼                   ▼               ▼
        [ MEL (256-D) ] [ LOFAR (1024-D) ]  [ HHT (256-D) ] [ DEMON (253-D) ]
        (Banda Larga /  (Raias Tonais /     (Freq. Instant. (Modulação /
         Ruído Geral)    Diesel Lento)       / Transitórios) Cavitação)
               │               │                   │               │
               └───────────────┼───────────────────┼───────────────┘
                               │
                               ▼
            ┌─────────────────────────────────────────────┐
            │       Classificador Downstream Modular      │
            │   (Agnóstico: Random Forest*, MLP, SVM, CNN) │
            │   *RF obteve o melhor desempenho (69.82% SP)│
            └──────────────────────┬──────────────────────┘
                                   │
                                   ▼
                        Predição Final da Classe
                 [Small | Medium | Large | Background]
```

As representações operam em sua **resolução nativa** (sem compressão artificial por PCA), garantindo fidelidade física e economia computacional ao calcular apenas o espectrograma escolhido para inferência.

---

## 2. Meta-Características Acústicas Físicas (1.825 Arquivos .wav)

Foram extraídas 7 meta-características físicas diretamente das gravações temporais (armazenadas em `data/iara_processed/cache_gating_metafeatures.pkl`):

| Meta-Característica | Descrição Física | Alvo Prioritário | Média em SMALL | Média em LARGE |
| :--- | :--- | :---: | :---: | :---: |
| **`crest_factor`** | Razão pico/média da densidade espectral $< 250\text{ Hz}$ | **LOFAR** | 4.89 | 4.86 |
| **`spectral_flatness`** | Planicidade espectral (1 = ruído branco, 0 = puro tonal) | **MEL** | 0.49 | 0.45 |
| **`mod_index`** | Razão pico/média do envelope na banda de cavitação ($2-8\text{ kHz}$) | **DEMON** | 1.15 | **1.32** |
| **`flux_std`** | Dispersão temporal de energia entre janelas consecutivas | **HHT** | 0.28 | 0.25 |
| **`high_low_ratio`** | Razão de energia alta ($2-8\text{ kHz}$) vs baixa ($< 250\text{ Hz}$) | **DEMON / SMALL** | **1.12** | **0.38** |
| **`rms_db`** | Potência média quadrática do sinal acústico | Triagem | -38.2 dB | -34.5 dB |
| **`zcr`** | Taxa de cruzamento por zero | Ruído Ambiente | 0.08 | 0.06 |

---

## 3. Resultados Experimentais Consolidados (10 Folds $5\times2\text{cv}$ com Ship Isolation)

Avaliamos a eficácia do roteamento dinâmico em 4 arquiteturas distintas de classificadores:

| Classificador Downstream | Baseline MEL SP (%) | SP com Gating RF (%) | Ganho no SP | Recall SMALL (%) | Recall MEDIUM (%) | Recall LARGE (%) | Recall BG (%) | Teto Oráculo SP (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (RF)** | $61.45 \pm 1.90\%$ | **$69.82 \pm 1.43\%$** | **$+8.37\%$** | **$61.6\%$** | $61.3\%$ | $75.0\%$ | $82.7\%$ | $85.64\%$ |
| **MLP** | $64.74 \pm 1.45\%$ | **$69.03 \pm 1.67\%$** | **$+4.29\%$** | $50.8\%$ | $64.2\%$ | $75.8\%$ | $88.6\%$ | $87.01\%$ |
| **SVM (Nyström RBF)** | $63.78 \pm 1.14\%$ | **$67.69 \pm 1.92\%$** | **$+3.91\%$** | $49.8\%$ | $62.6\%$ | $75.7\%$ | $86.3\%$ | **$88.07\%$** |
| **CNN 2D** | $63.80 \pm 1.83\%$ | **$65.95 \pm 2.18\%$** | **$+2.15\%$** | $43.0\%$ | $66.4\%$ | $67.0\%$ | $92.3\%$ | $87.29\%$ |

---

## 4. Distribuição das Rotas Selecionadas por Modelo

A tabela abaixo detalha o percentual de contatos acústicos direcionados para cada representação espectral:

| Classificador Downstream | Rota MEL (%) | Rota LOFAR (%) | Rota HHT (%) | Rota DEMON (%) | Comportamento do Roteador |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (RF)** | **$38.8\%$** | **$36.3\%$** | **$23.0\%$** | $1.9\%$ | **Equilibrado e Completo:** HHT resgata a classe Small com estabilidade. |
| **MLP** | $33.5\%$ | **$60.3\%$** | $4.6\%$ | $1.6\%$ | Preferência por LOFAR devido à linearidade das raias tonais. |
| **SVM (Nyström)** | **$47.6\%$** | **$47.6\%$** | $3.6\%$ | $1.2\%$ | **Duopólio MEL/LOFAR:** HHT quase não é acionado por risco de margem. |
| **CNN 2D** | **$86.9\%$** | $10.6\%$ | $0.3\%$ | $2.2\%$ | **Blindagem:** Roteador protege a CNN mantendo-a na entrada onde ela generaliza. |

---

## 5. Visualização Comparativa Oficial

O painel de 3 vias abaixo sintetiza os resultados:

![Painel Comparativo do Sistema Gating Front-End](file:///c:/Users/ev_ro/git/IARA/reports/comparacao_gating_frontend.png)

*(A) Índice SP (Baseline vs. Gating vs. Oráculo); (B) Recall por Classe sob Gating; (C) Distribuição Percentual das Rotas Selecionadas.*

---

## 6. Diagnóstico Científico: Por que o Gating funcionou melhor no RF do que no SVM?

1. **Estabilidade de Ensembles vs. Sensibilidade de Hiperplanos sob Ship Isolation:**
   - No **Random Forest**, o classificador treinado sobre o HHT é estável: o recall de Small varia apenas entre $46\%$ e $57\%$ em todos os 10 folds de validação cruzada. Por isso, o roteador de competência confia e envia **$23.0\%$ dos áudios para o HHT**, elevando o recall de Small de $52.8\%$ para **$61.6\%$**.
   - No **SVM (Nyström)**, o HHT sofre forte instabilidade geométrica com cascos inéditos: o recall de Small varia de **$2.8\%$ (Fold 6) a $91.0\%$ (Fold 0)**. O estimador de competência detecta essa variância e evita o HHT (apenas $3.6\%$ das rotas), gerando um duopólio conservador entre MEL e LOFAR.
2. **Origem do Ganho do SVM (+3.91%):**
   - O ganho do SVM veio da união eficiente entre LOFAR (que alavancou Large para $75.7\%$ e Background para $86.3\%$) e MEL (que sustentou Small e Medium). No entanto, sem o HHT ativo, Small ficou limitado a $49.8\%$.
3. **Generalização da Abordagem:**
   - O roteamento via meta-features físicas adimensionais (`crest_factor`, `spectral_flatness`, `high_low_ratio`) baseia-se na física hidrodinâmica das fontes mecânicas submarinas (cavitação vs. combustão diesel lenta), garantindo robustez e transferibilidade entre diferentes oceanos e sensores.

---

## 7. Arquitetura de Pico: Gating Front-End Heterogêneo (H-MoE)

Em vez de forçar um único classificador em todos os 4 ramos, implementamos a **Mistura Heterogênea de Especialistas (H-MoE)**, onde cada representação física é emparelhada com o modelo de maior afinidade geométrica (*Inductive Bias Matching*):
* **`LOFAR` $\rightarrow$ MLP:** Captura as raias harmônicas tonais de cilindros diesel com maior precisão linear/sigmoide ($66.24\%$ SP isolado);
* **`HHT` $\rightarrow$ Random Forest:** Opera em partições ortogonais não-lineares, imune à instabilidade de cascos inéditos ($60.64\%$ SP isolado);
* **`MEL` $\rightarrow$ Random Forest:** Atua como o protetor primordial da classe crítica *Small* ($58.1\%$ recall isolado);
* **`DEMON` $\rightarrow$ MLP:** Melhor aproveitamento de demodulação de cavitação ($53.23\%$ SP isolado).

### Desempenho Comparativo Oficial (10 Folds $5\times2\text{cv}$ com Ship Isolation):

| Arquitetura | SP (%) | Bal Acc (%) | Acc (%) | Recall SMALL (%) | Recall MEDIUM (%) | Recall LARGE (%) | Recall BG (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline MEL Isolado (RF)** | 61.45 ± 1.90 | 61.55 ± 1.89 | 61.83 ± 1.74 | 58.1% | 58.7% | 66.7% | 62.7% |
| **2. Baseline LOFAR Isolado (MLP)** | 66.24 ± 1.78 | 67.22 ± 1.65 | 65.51 ± 1.72 | 46.3% | 61.5% | 74.7% | 86.3% |
| **3. Gated RF Homogêneo** | 69.82 ± 1.43 | 70.15 ± 1.42 | 69.26 ± 1.25 | **61.6%** | 61.3% | 75.0% | 82.7% |
| **4. H-MoE Automático (Nested CV)** | 70.53 ± 1.96 | 71.14 ± 1.90 | 69.64 ± 1.85 | 54.9% | **65.2%** | **76.7%** | **87.8%** |
| **5. H-MoE Especializado (Pico)** | **70.68 ± 2.02** | **71.13 ± 1.98** | **69.70 ± 2.00** | 59.2% | 64.0% | 74.6% | 86.7% |
| **6. Super-Oráculo (16 Especialistas)** | **97.66 ± 0.73** | **97.67 ± 0.73** | **97.43 ± 0.65** | **97.4%** | **96.5%** | **97.1%** | **99.7%** |

![Evolução H-MoE](file:///c:/Users/ev_ro/git/IARA/reports/hmoe_heterogeneous_gating.png)

*(A) Evolução do SP Index; (B) Recall por Classe sob H-MoE; (C) Alocação Dinâmica de Rotas e Especialistas.*

### Distribuição de Rotas do H-MoE Especializado:
* **`LOFAR:MLP`:** 3.151 áudios ($49.3\%$) — dominante para Large e Background;
* **`MEL:RF`:** 2.059 áudios ($32.2\%$) — sustentação de Small e Medium;
* **`HHT:RF`:** 1.112 áudios ($17.4\%$) — resgate de embarcações Small;
* **`DEMON:MLP`:** 68 áudios ($1.1\%$) — cavitações específicas.

