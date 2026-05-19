# Análise de Acurácia por Classe Acústica - SVM (m=2000)

Este documento apresenta a análise de acurácia discriminada por classe acústica obtida através do classificador **SVM aproximado via Kernel RBF Nyström (com m=2000 componentes)** na representação espectral **log-Melgram (MEL)**.

A avaliação foi realizada sob a estratégia de **Votação por Áudio (by_audio)** empregando uma validação cruzada de **10 folds** robusta por arquivo de áudio (evitando qualquer vazamento de dados espectrais entre treino e teste).

---

## 📊 1. Matriz de Confusão Consolidada (10 Folds)

A tabela abaixo cruza as classes reais das gravações de áudio com as predições majoritárias efetuadas pelo classificador:

| CLASSE REAL \ PREVISTA | 🚤 SMALL | 🚢 MEDIUM | ⚓ LARGE | 🌊 BACKGROUND | TOTAL DE TESTES |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **🚤 SMALL** (Pequeno) | **`706`** | 504 | 378 | 192 | **1780** |
| **🚢 MEDIUM** (Médio) | 293 | **`948`** | 218 | 31 | **1490** |
| **⚓ LARGE** (Grande) | 214 | 327 | **`1531`** | 68 | **2140** |
| **🌊 BACKGROUND** (Silêncio) | 63 | 46 | 58 | **`813`** | **980** |

---

## 🎯 2. Desempenho e Acurácia (Recall) por Classe

* 🌊 **`BACKGROUND` (Ruído de Fundo):** **`83.0%`** de acerto *(813 de 980 corretos)*
* ⚓ **`LARGE` (Navio Grande):** **`71.5%`** de acerto *(1531 de 2140 corretos)*
* 🚢 **`MEDIUM` (Navio Médio):** **`63.6%`** de acerto *(948 de 1490 corretos)*
* 🚤 **`SMALL` (Navio Pequeno):** **`39.7%`** de acerto *(706 de 1780 corretos)*

---

## 🧠 3. Diagnóstico Físico e Acústico das Classes

### A. O Sucesso no Silêncio e Grandes Navios (`BACKGROUND` e `LARGE`)
* **BACKGROUND (83.0%):** O modelo demonstra excelente capacidade de discernir o ruído natural do oceano (chuva, ondas, biologia marinha) de assinaturas antrópicas. A ausência de componentes harmônicas contínuas é um padrão linear e limpo, facilmente separado pelo classificador.
* **LARGE (71.5%):** Navios grandes (cargueiros, petroleiros com comprimento $\ge 100\text{m}$) possuem motores diesel lentos acoplados a hélices colossais. Esse sistema gera uma assinatura de cavitação de baixa frequência (graves/infra-sons) extremamente estável e potente na água. Essas raias espectrais contínuas em baixas frequências são padrões geométricos muito nítidos e fáceis de classificar.

### B. O Gargalo dos Navios Pequenos (`SMALL` - 39.7%)
Esta classe é a principal responsável por segurar a acurácia global do modelo na faixa de ~64.5%. A física acústica explica as principais fontes de confusão:
1. **Confusão com BACKGROUND (192 erros):** Embarcações de pequeno porte (lanchas rápidas, barcos de pesca pequenos de fibra ou madeira com comprimento $< 50\text{m}$) produzem baixíssima potência acústica. Em muitos trechos de gravação, o sinal emitido por essas embarcações é completamente mascarado pelo ruído ambiente do mar (baixo SNR - Relação Sinal-Ruído), fazendo com que o classificador o confunda com silêncio.
2. **Confusão com MEDIUM e LARGE (504 + 378 erros):** Pequenos barcos de pesca usam motores diesel comuns que compartilham assinaturas harmônicas (frequências de ignição dos cilindros) muito parecidas com rebocadores portuários (`MEDIUM`). Devido à variação de rotação (efeito doppler em barcos rápidos) e à fraca amplitude do sinal, o SVM RBF aproxima as fronteiras de decisão de forma sensível, categorizando-os erroneamente em classes superiores de tamanho.

---

## 💡 4. Plano de Ação Metodológico para Melhoria da Acurácia

Para romper a barreira dos 65% de acurácia global e empurrar o classificador para a faixa de **70% a 75%**, os esforços de ajuste fino devem focar na classe `SMALL`:
* **Ajuste Fino do Gamma ($\gamma$):** Reduzir ou ajustar o parâmetro gamma do Kernel RBF RBF-SVM pode suavizar a sensibilidade do classificador a pontos fracos, ajudando a delinear melhor a fronteira sutil da assinatura de barcos pequenos sem que ela seja absorvida pelas classes maiores.
* **Regularização ($C$):** O teste ativo com regularizações diferentes nos dará o equilíbrio ideal para ignorar pequenos picos de ruído nas amostras de barcos pequenos, promovendo maior estabilidade no teste.
* **Filtros de SNR (Relação Sinal-Ruído):** Uma possibilidade futura é aplicar um limiar de energia espectral para destacar as raias harmônicas dos pequenos navios antes da projeção de Nyström.
