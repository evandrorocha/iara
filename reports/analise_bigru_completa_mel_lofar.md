# Análise Comparativa Completa: BiGRU com Atenção Temporal (MEL vs LOFAR e Fusões Multimodais)

> **Correção de 23/10/2026 — extração HHT.** O cache HHT usado neste documento misturava duas configurações de EMD (`MAX_ITERATION` ≥ 100 em 571 gravações e 30 nas demais; nenhuma gravação BACKGROUND no primeiro grupo). O HHT foi re-extraído com configuração única (8 IMFs, `MAX_ITERATION=30`) e todos os experimentos HHT foram retreinados; a auditoria e o antes × depois estão no `notebooks/28_correcao_extracao_hht.ipynb`. Os números HHT abaixo são **anteriores à correção** (e alguns vêm de execuções ainda mais antigas que as predições em disco). Valores atuais, recalculados das predições: BiGRU HHT 4c SP 58,18 e 3c 57,65; MEL + HHT 4c 63,95 e 3c 62,69; LOFAR + HHT 4c 63,99 e 3c 60,40 (nenhuma mudança significativa no F-test 5x2cv). A BiGRU não tem semente fixa.

**Protocolo:** Validação Cruzada $5\times2\text{cv}$ Oficial (10 Folds com isolamento estrito de navios)  
**Aceleração:** NVIDIA GeForce RTX 5060 Ti (PyTorch / CUDA)  
**Arquitetura:** BiGRU bidirecional (128 unidades ocultas, 2 camadas) + Atenção Temporal + Agregação por Votação por Arquivo de Áudio ($L=15$, $stride=5$)

---

## 1. Tabela Geral de Resultados — 4 Classes e 3 Classes

| Classes | Representação | Dimensão | SP (%) | Bal Acc (%) | SMALL Recall | MEDIUM Recall | LARGE Recall | BG Recall | Tempo (10f) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **4 Classes** | DEMON | 253-D | 52.15% ± 1.13% | 53.15% ± 1.04% | 33.0% | 50.9% | 58.9% | 69.8% | 59.5s |
| **4 Classes** | HHT | 256-D | 55.93% ± 1.43% | 57.38% ± 1.23% | 32.8% | 56.8% | 60.5% | 79.5% | 55.7s |
| **4 Classes** | MEL | 256-D | 61.19% ± 1.17% | 62.27% ± 1.30% | 40.2% | 60.6% | 65.0% | 83.3% | 54.1s |
| **4 Classes** | **FUSÃO MEL + DEMON** | 509-D | 61.29% ± 1.05% | 62.04% ± 1.07% | 44.0% | 58.5% | 65.3% | 80.4% | 64.2s |
| **4 Classes** | **FUSÃO MEL + HHT** | 512-D | **62.35% ± 1.14%** | **63.17% ± 1.08%** | 43.5% | **61.2%** | **67.4%** | 80.6% | 63.6s |
| **4 Classes** | **LOFAR** | 1024-D | 60.03% ± 0.96% | 60.55% ± 0.94% | **51.8%** | 49.4% | 65.7% | 75.2% | 74.5s |
| **4 Classes** | **FUSÃO LOFAR + DEMON** | 1277-D | **61.47% ± 1.03%** | **61.91% ± 0.97%** | **54.3%** | 50.9% | 66.2% | 76.3% | 77.7s |
| **4 Classes** | **FUSÃO LOFAR + HHT** | 1280-D | **61.33% ± 1.28%** | **61.82% ± 1.33%** | **53.3%** | 50.6% | 65.8% | 77.6% | 74.4s |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **3 Classes** | DEMON | 253-D | 52.87% ± 1.19% | 53.14% ± 1.25% | 44.9% | 53.9% | 60.6% | — | 53.1s |
| **3 Classes** | HHT | 256-D | 55.61% ± 1.43% | 55.99% ± 1.52% | 44.4% | 60.7% | 62.8% | — | 54.6s |
| **3 Classes** | MEL | 256-D | 58.97% ± 1.42% | 59.29% ± 1.45% | 49.9% | 59.5% | 68.5% | — | 51.9s |
| **3 Classes** | **FUSÃO MEL + DEMON** | 509-D | 59.12% ± 1.05% | 59.35% ± 0.98% | 51.8% | 59.8% | 66.4% | — | 58.9s |
| **3 Classes** | **FUSÃO MEL + HHT** | 512-D | **59.79% ± 1.43%** | **60.06% ± 1.38%** | 50.5% | **60.7%** | **69.0%** | — | 60.3s |
| **3 Classes** | **LOFAR** | 1024-D | 56.58% ± 1.46% | 56.81% ± 1.40% | **52.9%** | 50.8% | 66.7% | — | 63.1s |
| **3 Classes** | **FUSÃO LOFAR + DEMON** | 1277-D | **58.04% ± 1.47%** | **58.26% ± 1.44%** | **55.7%** | 51.8% | 67.3% | — | 72.3s |
| **3 Classes** | **FUSÃO LOFAR + HHT** | 1280-D | **57.83% ± 1.08%** | **58.06% ± 1.04%** | **55.9%** | 50.9% | 67.4% | — | 68.5s |

---

## 2. Gráfico Consolidado

![Comparação Completa BiGRU](reports/comparacao_completa_bigru_multiclasse.png)

---

## 3. Principais Conclusões Científicas

1. **O Teto das Redes Neurais com Memória no IARA ($58\% - 62\%$):**
   - No repositório, o modelo baseline convolucional-recorrente `CRNN 2D LOFAR` treinado anteriormente obteve **SP = 58.01%** (por janela) e **60.47%** (por arquivo).
   - O BiGRU com LOFAR puro obteve **SP = 60.03% ± 0.96%**, subindo para **61.47% ± 1.03%** na fusão com DEMON e **62.35% ± 1.14%** na fusão MEL + HHT.
   - Isso demonstra de forma empírica que **arquiteturas temporais (BiGRU / CRNN) encontram um teto estrutural em torno de ~62% no IARA**.
   - *Causa física:* O protocolo do IARA impõe **isolamento de navios ($5\times2\text{cv}$)**. A dependência temporal aprendida no conjunto de treino captura a assinatura cinemática/aceleração específica dos navios de treino; ao ser testada em navios nunca antes vistos, essa dinâmica temporal falha em generalizar com a mesma eficácia que os hiperplanos estáticos do SVM e da MLP.

2. **O Salto Histórico na Classe SMALL proporcionado por LOFAR e DEMON:**
   - O grande gargalo de reconhecimento sempre foi a classe `SMALL`:
     - Com **MEL puro no BiGRU**: Recall de `SMALL` foi de apenas **40.2%** (4C).
     - Com **LOFAR puro no BiGRU**: Recall de `SMALL` saltou para **51.8%** (+11.6 p.p.).
     - Com **FUSÃO LOFAR + DEMON no BiGRU**: Recall de `SMALL` bateu **54.3%** (4C) e **55.7%** (3C).
     - Com **FUSÃO LOFAR + HHT no BiGRU**: Recall de `SMALL` atingiu **53.3%** (4C) e **55.9%** (3C).

3. **Por que o SP do LOFAR no BiGRU ficou em 60%-61% se o Small foi tão alto?**
   - Enquanto o MEL favorece `MEDIUM` (60.6%) e `BACKGROUND` (83.3%), o LOFAR no BiGRU sofreu especificamente na classe `MEDIUM` (caindo para 49.4% - 50.9%).
   - Como o índice SP penaliza desequilíbrios entre as classes através da média geométrica, a queda no recall de Medium equilibrou a pontuação geral em torno de 60%-61.5%.
