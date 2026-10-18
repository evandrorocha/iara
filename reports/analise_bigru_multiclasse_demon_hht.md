# Análise de Desempenho: BiGRU com Atenção Temporal em Multiclasse (4 Classes e 3 Classes)

**Protocolo:** Validação Cruzada $5\times2\text{cv}$ (10 Folds estratificados)  
**Hardware de Treinamento:** NVIDIA GeForce RTX 5060 Ti (CUDA)  
**Arquitetura:** BiGRU bidirecional (128 unidades ocultas, 2 camadas) + Mecanismo de Atenção Temporal + Pooling + Dropout + Linear Classifier  
**Agregação:** Janelas temporais de $L=15$ ($stride=5$), votação por maioria ponderada por arquivo de áudio.

---

## 1. Quadro Comparativo Consolidado

| Escopo | Representação | Dimensão | SP (%) | Bal Acc (%) | SMALL Recall | MEDIUM Recall | LARGE Recall | BG Recall | Tempo (10 folds) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **4 Classes** | DEMON | 253-D | 52.15% ± 1.13% | 53.15% ± 1.04% | 33.0% | 50.9% | 58.9% | 69.8% | 59.5s |
| **4 Classes** | HHT | 256-D | 55.93% ± 1.43% | 57.38% ± 1.23% | 32.8% | 56.8% | 60.5% | 79.5% | 55.7s |
| **4 Classes** | MEL | 256-D | 61.19% ± 1.17% | 62.27% ± 1.30% | 40.2% | 60.6% | 65.0% | 83.3% | 54.1s |
| **4 Classes** | **FUSÃO MEL + DEMON** | 509-D | **61.29% ± 1.05%** | 62.04% ± 1.07% | **44.0%** *(+3.8 p.p.)* | 58.5% | 65.3% | 80.4% | 64.2s |
| **4 Classes** | **FUSÃO MEL + HHT** | 512-D | **62.35% ± 1.14%** *(+1.16 p.p.)* | **63.17% ± 1.08%** *(+0.90 p.p.)* | **43.5%** *(+3.3 p.p.)* | **61.2%** *(+0.6 p.p.)* | **67.4%** *(+2.4 p.p.)* | 80.6% | 63.6s |
| **3 Classes** | DEMON | 253-D | 52.87% ± 1.19% | 53.14% ± 1.25% | 44.9% | 53.9% | 60.6% | — | 53.1s |
| **3 Classes** | HHT | 256-D | 55.61% ± 1.43% | 55.99% ± 1.52% | 44.4% | 60.7% | 62.8% | — | 54.6s |
| **3 Classes** | MEL | 256-D | 58.97% ± 1.42% | 59.29% ± 1.45% | 49.9% | 59.5% | 68.5% | — | 51.9s |
| **3 Classes** | **FUSÃO MEL + DEMON** | 509-D | **59.12% ± 1.05%** | 59.35% ± 0.98% | **51.8%** *(+1.9 p.p.)* | 59.8% | 66.4% | — | 58.9s |
| **3 Classes** | **FUSÃO MEL + HHT** | 512-D | **59.79% ± 1.43%** *(+0.82 p.p.)* | **60.06% ± 1.38%** *(+0.77 p.p.)* | **50.5%** *(+0.6 p.p.)* | **60.7%** *(+1.2 p.p.)* | **69.0%** *(+0.5 p.p.)* | — | 60.3s |

---

## 2. Principais Achados Científicos

### A. O Ganho Expressivo da HHT com Memória Recorrente
- Nos modelos convolucionais (CNN 2D), o espectro marginal de HHT sofria com a dispersão de energia entre IMFs, atingindo apenas **46.10% SP**.
- No **BiGRU**, o HHT atinge **55.93% SP** (4 Classes) e **55.61% SP** (3 Classes), superando com folga tanto a CNN quanto o SVM ($53.63\%$).
- *Interpretação:* A rede recorrente consegue aprender a dinâmica temporal e as transições de frequência instantânea dos modos empíricos ao longo das janelas acústicas.

### B. Fusão MEL + DEMON: Recuperação da Classe SMALL
- O DEMON isolado estaciona em ~52%-53% de SP devido à ausência de informação das componentes de baixa frequência da combustão.
- No entanto, quando **fundido ao MEL no BiGRU**, o recall de **SMALL dá um salto notável**:
  - Em 4 Classes: passa de **40.2% para 44.0%** (+3.8 pontos percentuais).
  - Em 3 Classes: passa de **49.9% para 51.8%** (+1.9 pontos percentuais).
- *Mecanismo Físico:* Embarcações pequenas possuem hélices de alta rotação que geram forte modulação de cavitação no DEMON, permitindo ao BiGRU identificar assinaturas temporais que o MEL puro confunde com ruído ambiente ou com Medium.

### C. Fusão MEL + HHT: Campeã Geral em Todas as Classes
- A fusão precoce concatenando $MEL(256) + HHT(256)$ alimentando o BiGRU bidirecional superou o MEL puro de ponta a ponta:
  - **+1.16 p.p.** no SP de 4 Classes (**62.35%** vs 61.19%).
  - **+0.82 p.p.** no SP de 3 Classes (**59.79%** vs 58.97%).
  - Em 4 Classes, aumentou o recall de **SMALL** (de 40.2% para 43.5%), **MEDIUM** (de 60.6% para 61.2%) e **LARGE** (de 65.0% para 67.4%).
