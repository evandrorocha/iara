import json
import os

notebook_path = 'notebooks/24_sistema_gating_frontend.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.loads(f.read())

# Cell 0: Introdução e Motivação
cell_0_md = """# Sistema de Gating Front-End (Roteamento Dinâmico de Entradas) – Dataset IARA
### Notebook 24: Arquitetura Multimodal Agnóstica ao Modelo Classificador com Suporte a 4 Representações

Neste notebook, investigamos e validamos a arquitetura de **Gating Front-End (Multiplexador Dinâmico de Entradas)** no problema de reconhecimento de alvos acústicos submarinos passivos (*Underwater Acoustic Target Recognition* - UATR) sobre o dataset de grande escala **IARA** (*Silva et al.*, IEEE Access 2025).

### Princípios da Arquitetura de Roteamento:
1. **O Foco é o Processo de Roteamento da Entrada:** O roteador avalia as características físicas do sinal acústico bruto e seleciona a melhor representação espectral (**MEL**, **LOFAR**, **HHT** ou **DEMON**) *antes* de executar transformações pesadas.
2. **Modelo Downstream Agnóstico e Modular:** O classificador final não é fixo. A arquitetura suporta qualquer família de modelo (Random Forest, MLP, SVM Nyström ou CNN 2D). Nos experimentos comparativos, o **Random Forest** obteve o melhor desempenho global (**SP de 69.82%** e **61.6% de recall em Small**), mas a engenharia do sistema permite plugar qualquer outro modelo de interesse.
3. **Preservação da Resolução Física sem Compressão Artificial:** Cada representação opera em sua dimensão nativa (MEL 256-D, LOFAR 1024-D, HHT 256-D, DEMON 253-D), eliminando compressões destrutivas por PCA.
4. **Eliminação de Cascatas Hierárquicas:** Ao tomar a decisão na entrada, o sistema evita o acúmulo de latência e a propagação de erros de predição sequenciais.
5. **Economia Computacional:** Como apenas o espectrograma selecionado é gerado para cada contato sonar, economiza-se mais de 70% do custo computacional em tempo real.

Todos os testes foram executados sob o protocolo oficial de **validação cruzada $5\\times2\\text{cv}$ (10 folds estratificados)** com isolamento estrito de navios (`Ship ID`).
"""
nb['cells'][0]['source'] = [line + '\n' for line in cell_0_md.split('\n')]

# Cell 2: Diagrama da Arquitetura com HHT e Modelo Agnóstico
cell_2_md = """## 1. Arquitetura do Sistema Gating Front-End

Abaixo apresentamos o fluxo de processamento do sistema. O módulo `Acoustic Front-End Gating` atua como um multiplexador dinâmico na entrada, roteando o áudio entre as 4 representações físicas fundamentais:

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
                                           │ Seleção Dinâmica da Entrada
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

### Papel de Cada Ramo Espectral na Física Acústica:
* **`MEL` (256-D):** Filtro psicoacústico de banda larga; excelente como representação padrão de ruído hidrodinâmico contínuo.
* **`LOFAR` (1024-D):** Banda estreita em baixa frequência ($< 250\\text{ Hz}$); captura as raias harmônicas tonais de cilindros diesel de navios de grande porte (*Large*).
* **`HHT` (256-D):** Transformada de Hilbert-Huang via Decomposição em Modos Empíricos (EMD); extrai frequências instantâneas de eventos não-estacionários e pulsos de alta rotação (*Small*).
* **`DEMON` (253-D):** Demodulação de envoltória na banda de cavitação ($2 - 8\\text{ kHz}$); mede a frequência de rotação do eixo e batimento de pás de hélice.
"""
nb['cells'][2]['source'] = [line + '\n' for line in cell_2_md.split('\n')]

# Cell 9: Conclusões científicas alinhadas à modularidade
cell_9_md = """## 5. Conclusões Científicas e Recomendações de Engenharia

### Principais Conclusões:
1. **O Mecanismo de Roteamento na Entrada é o Diferencial:**
   * O multiplexador dinâmico no front-end decide a rota com base em propriedades físicas do sinal bruto em microssegundos.
   * Evita a geração redundante de múltiplos espectrogramas pesados e elimina cascatas de classificação que propagam erros.
2. **Modularidade do Classificador Downstream:**
   * O sistema é agnóstico ao modelo: o usuário ou engenheiro pode plugar **Random Forest**, **MLP**, **SVM** ou **CNN 2D**.
   * Entre as 4 famílias testadas, o **Random Forest** alcançou o **melhor índice SP geral ($69.82\% \\pm 1.43\%$)** e o maior salto de desempenho (**$+8.37\%$** sobre o baseline MEL), resgatando a classe crítica *Small* de $52.8\\%$ para **$61.6\\%$**.
3. **Por que o Random Forest se destacou:**
   * O RF aproveitou de forma plena o ramo **HHT ($23.0\\%$ das rotas)**, mantendo recall consistente em Small ($46\\% - 57\\%$) em todos os folds sob *Ship Isolation*.
   * Em contrapartida, modelos como o SVM apresentaram forte instabilidade de hiperplano no HHT para cascos inéditos (recall de Small oscilando de $2.8\\%$ a $91.0\\%$), levando o roteador a adotar uma postura conservadora (duopólio MEL/LOFAR com apenas $3.6\\%$ HHT, atingindo SP de $67.69\\%$).
4. **Generalização da Abordagem:**
   * Como o roteador opera sobre meta-características adimensionais e de razão relativa (`high_low_ratio`, `crest_factor`, `spectral_flatness`), o processo de roteamento baseia-se em princípios acústicos universais (cavitação de alta rotação vs. combustão diesel lenta), garantindo robustez para novos cascos e sensores.
"""
nb['cells'][9]['source'] = [line + '\n' for line in cell_9_md.split('\n')]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook 24 atualizado com arquitetura completa (HHT incluído) e modelo agnóstico!")
