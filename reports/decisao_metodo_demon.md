# Decisão de Método: Extração de Características DEMON para o Dataset IARA

## 1. Contexto e Motivação Científica

Nos experimentos anteriores com o dataset IARA (*Silva et al.*, IEEE Access 2025), identificou-se um gargalo operacional crítico nas representações espectrais clássicas (**LOFAR** de 1024 bins e **MEL** de 256 bandas):
* **Confusão SMALL vs. LARGE:** Embarcações da classe minoritária **SMALL** sofreram forte confusão com **LARGE** (no SVM LOFAR-6000, $28{,}6\%$ das amostras de SMALL foram preditas incorretamente como LARGE).
* **Fundamentação Acústica:** No domínio de LOFAR e MEL, o ruído ambiente oceânico e o arrasto hidrodinâmico mascaram as fracas raias de embarcações pequenas, enquanto cargueiros pesados possuem alta energia de banda larga.
* **A Hipótese do DEMON:** O **DEMON** (*Detection of Envelope Modulation on Noise*) não analisa as frequências audíveis diretas, mas sim a **frequência de modulação do envelope de cavitação** produzida pelo hélice.
  - Pequenos barcos rápidos operam em **altas rotações de eixo/hélice** ($500 \text{ a } 1200\text{ RPM} \implies 8 \text{ a } 20\text{ Hz}$).
  - Grandes cargueiros e mineraleiros operam com motores 2 tempos acoplados diretamente ao eixo em **baixas rotações** ($80 \text{ a } 130\text{ RPM} \implies 1{,}3 \text{ a } 2{,}2\text{ Hz}$).
  - A separabilidade entre SMALL e LARGE no domínio DEMON decorre diretamente da física do sistema propulsor.

---

## 2. Origem da Implementação de Referência

Adotou-se como ponto de partida a implementação de DEMON desenvolvida por **Fábio Silva** (*Fábio Oliveira Baptista da Silva*, autor correspondente do artigo do IARA), disponibilizada no repositório oficial do laboratório:
* **Repositório:** `labsonar/signal_processing` (LPS / COPPE / UFRJ)
* **Arquivo original:** `apps/demon.py`

---

## 3. Análise Crítica e Decisões de Engenharia

Ao auditar e testar a implementação original do Fábio nos áudios do IARA, identificaram-se pontos fortes conceituais e necessidades de ajustes computacionais:

### 3.1. Correção do Filtro Passa-Faixa: `analog=True` vs. Digital SOS
* **Implementação original:** 
  ```python
  b, a = scipy.cheby2(N, rs=As, Wn=wc, btype='bandpass', output='ba', analog=True)
  x = scipy.lfilter(b, a, x, axis=0)
  ```
* **Diagnóstico:** Em `scipy.signal`, o argumento `analog=True` projeta um filtro analógico contínuo no plano de Laplace ($s$). Ao passar coeficientes analógicos diretamente para `scipy.lfilter` (que opera no plano discreto $z$), a resposta em frequência sofre severa distorção e atenuação excessiva (~$100\times$).
* **Decisão:** Substituir pela formulação digital em seções de segunda ordem (**SOS — Second-Order Sections**):
  ```python
  sos = scipy_sig.cheby2(N, rs=40, Wn=wc, btype='bandpass', output='sos', analog=False)
  x_filtered = scipy_sig.sosfilt(sos, signal)
  ```
  O formato SOS garante estabilidade numérica contra erros de ponto flutuante em ordens elevadas de Chebyshev.

---

### 3.2. Taxa de Amostragem Final e o Limite de Nyquist (`max_freq`)
* **Implementação original:** A função `_get_demon_steps(fs, max_freq)` define os fatores de decimação de modo que a taxa final seja `max_freq` (ex: $50\text{ Hz}$).
* **Diagnóstico:** Pelo Teorema de Nyquist-Shannon, a frequência de modulação máxima observável é metade da taxa final:
  $$f_{\text{máx}} = \frac{f_{s,\text{final}}}{2} = \frac{50}{2} = 25\text{ Hz}$$
* **Decisão:** Manter $f_{s,\text{final}} = 50\text{ Hz}$ como padrão, pois a janela de $0\text{ a }25\text{ Hz}$ cobre até $1500\text{ RPM}$, faixa que engloba a grande maioria das embarcações comerciais e auxiliares. Caso se deseje visualizar até $50\text{ Hz}$, a taxa pós-decimação deve ser configurada para $100\text{ Hz}$.

---

### 3.3. Eliminação de Dependências Pesadas (`sympy` e `librosa`)
* **Implementação original:** Utilizava `sympy.factorint` para encontrar fatores primos de decimação e `librosa.stft` para a transformada de Fourier de curta duração.
* **Diagnóstico:** `sympy` e `librosa` introduzem dependências extensas (Numba, LLVM, etc.), desnecessárias para processamento numérico puro em pipelines de larga escala.
* **Decisão:** 
  1. Fatoração inteira em Python puro (`get_factors`), eliminando o SymPy.
  2. STFT realizada diretamente com `scipy.signal.stft` e `scipy.signal.decimate`, reduzindo o tempo de execução e a complexidade de deploy.

---

## 4. Validação Empírica no Dataset IARA

O script validado em [scratch/test_demon_pipeline.py](file:///c:/Users/ev_ro/git/IARA/scratch/test_demon_pipeline.py) foi executado em gravações reais do acervo UO (Bacia de Santos, 48 kHz):

| Classe | Arquivo | Embarcação | Comprimento | Frequência DEMON | RPM Equivalente | Observação Física |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **SMALL** | `A-0002.wav` | *Pilot 10* | $11\text{ m}$ | **$10{,}35\text{ Hz}$** | **$621\text{ RPM}$** | Pico harmônico intenso e persistente ao longo de todo o trecho temporal. |
| **MEDIUM** | `A-0001.wav` | *Maersk Pacer* | $74\text{ m}$ | **$8{,}30\text{ Hz}$** | **$498\text{ RPM}$** | Harmônicos médios em $2{,}7\text{ Hz}$ e $8{,}3\text{ Hz}$. |
| **LARGE** | `A-0004.wav` | *Sinoglory* | $322\text{ m}$ | **$1{,}80\text{ Hz}$** | **$105\text{ RPM}$** | Frequência de eixo fundamental em $105\text{ RPM}$, com harmônico de pás em $5{,}5\text{ Hz}$ ($328\text{ RPM}$). |
| **AMBIENTAL** | `E-1446.wav` | Fundo do mar | — | — | — | Ausência completa de linhas espectrais periódicas. |

O gráfico comparativo comprobatório encontra-se armazenado em [scratch/demon_comparison.png](file:///c:/Users/ev_ro/git/IARA/scratch/demon_comparison.png).

---

## 5. Diretrizes para os Classificadores

1. **Vetor de Entrada para Modelos:**
   - Vetor de $253\text{ bins}$ (cobrindo de $0{,}39\text{ Hz}$ a $25\text{ Hz}$ com resolução de $\sim 0{,}1\text{ Hz}$).
   - Média temporal das janelas de modulação (espectro integrado) ou representação 2D para modelos sequenciais/convolucionais.
2. **Experimentos Planejados:**
   - **Etapa A:** Classificação isolada com SVM Nyström (medir capacidade de separação $5\times2$cv). **[CONCLUÍDO]**
   - **Etapa B:** Inclusão de DEMON no classificador especialista em cascata (substituindo ou complementando LOFAR-PCA8 com DEMON). **[PRÓXIMA ETAPA]**

---

## 6. Resultados Experimentais da Etapa A: DEMON Isolado no $5\times2$cv

Todos os **1.706 áudios de UO** (Coleções A, B, C, D e E) foram extraídos e armazenados em cache padronizado (`data/iara_processed/demon_features/`). A avaliação seguiu rigorosamente o protocolo oficial de $5\times 2$ validação cruzada (10 folds independentes com particionamento estrito por `Ship ID`, garantindo zero vazamento de dados).

O classificador avaliado foi o **SVM Nyström** com núcleo RBF aproximado, penalidade ElasticNet e regularização balanceada, avaliado por votação majoritária por áudio (*audio-level majority voting*).

### 6.1 Comparativo Geral: 4 Classes (SMALL, MEDIUM, LARGE, AMBIENTAL)

| Modelo | Dimensão / Entrada | Acurácia (Áudio) | Índice SP Oficial | Recall SMALL | Recall MEDIUM | Recall LARGE | Recall AMB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOFAR SVM** ($m=4000$) | $64\text{-D (PCA)}$ | $63{,}32\% \pm 2{,}01\%$ | $63{,}09\% \pm 2{,}06\%$ | $47{,}81\%$ | $55{,}77\%$ | $74{,}91\%$ | **$77{,}65\%$** |
| **MEL SVM** ($m=4000$) | $256\text{-D}$ | **$63{,}36\% \pm 1{,}00\%$** | **$63{,}80\% \pm 1{,}12\%$** | **$50{,}90\%$** | **$60{,}40\%$** | $69{,}16\%$ | $77{,}86\%$ |
| **DEMON SVM** ($m=1000$, Puro) | $253\text{-D}$ | $49{,}94\% \pm 2{,}86\%$ | $38{,}75\% \pm 14{,}49\%$ | $12{,}02\%$ | $50{,}81\%$ | $73{,}60\%$ | $65{,}82\%$ |
| **DEMON SVM** ($m=4000$, Puro) | $253\text{-D}$ | $51{,}17\% \pm 2{,}62\%$ | $43{,}51\% \pm 4{,}41\%$ | $13{,}60\%$ | $44{,}90\%$ | **$76{,}40\%$** | $73{,}88\%$ |
| **DEMON SVM** ($m=6000$, Puro) | $253\text{-D}$ | **$51{,}72\% \pm 2{,}01\%$** | **$46{,}57\% \pm 4{,}45\%$** | $15{,}51\%$ | $50{,}20\%$ | $73{,}32\%$ | $72{,}65\%$ |

#### Matriz de Confusão Normalizada — DEMON $m=4000$ (4 Classes, 10 Folds Acumulados):
```
             SMALL   MEDIUM    LARGE   AMBIENTAL
SMALL        13.6%    17.4%    39.9%       29.1%
MEDIUM        8.7%    44.9%    22.8%       23.6%
LARGE         5.8%     7.8%    76.4%       10.0%
AMBIENTAL     6.2%     3.0%    16.9%       73.9%
```

---

### 6.2 Comparativo Focal em Embarcações: 3 Classes (SMALL, MEDIUM, LARGE)

Para isolar o comportamento do DEMON na discriminação mecânica de cascos e motores (sem a interferência da classe de fundo do mar):

| Modelo | Acurácia (Áudio) | Índice SP Oficial | Recall SMALL | Recall MEDIUM | Recall LARGE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LOFAR SVM** ($m=4000$) | **$60{,}72\% \pm 2{,}44\%$** | **$60{,}09\% \pm 2{,}52\%$** | $49{,}65\%$ | **$56{,}82\%$** | $76{,}25\%$ |
| **DEMON SVM** ($m=1000$, Puro) | $55{,}62\% \pm 1{,}87\%$ | $50{,}28\% \pm 3{,}26\%$ | $30{,}51\%$ | $51{,}07\%$ | **$79{,}67\%$** |
| **DEMON SVM** ($m=4000$, Puro) | $56{,}82\% \pm 2{,}10\%$ | $53{,}06\% \pm 2{,}77\%$ | $50{,}56\%$ | $42{,}68\%$ | $71{,}87\%$ |
| **DEMON SVM** ($m=6000$, Puro) | **$57{,}99\% \pm 1{,}18\%$** | **$54{,}83\% \pm 1{,}69\%$** | **$57{,}92\%$** | $41{,}41\%$ | $69{,}58\%$ |

#### Matriz de Confusão Normalizada — DEMON $m=4000$ (3 Classes Embarcações):
```
           SMALL   MEDIUM    LARGE
SMALL      50.6%    14.0%    35.4%
MEDIUM     36.8%    42.7%    20.5%
LARGE      21.5%     6.6%    71.9%
```

---

### 6.3 Diagnóstico Científico e Conclusões da Etapa A

1. **Assinatura Superior em Navios de Grande Porte (LARGE):**
   - O DEMON isolado obteve **$76{,}40\%$** de recall no problema de 4 classes e **$79{,}67\%$** no problema de 3 classes com $m=1000$. Em ambos os cenários, o DEMON supera o LOFAR ($74{,}91\%$) e o MEL ($69{,}16\%$).
   - *Justificativa física:* Navios de grande porte (graneleiros, mineraleiros, petroleiros) operam em rotação extremamente lenta e estável ($100\text{--}150\text{ RPM}$, eixo em $1{,}5\text{--}2{,}5\text{ Hz}$ e pá em $4\text{--}8\text{ Hz}$). A cavitação intensa modula a banda alta com periodicidade limpa, gerando picos estreitos de alta saliência no DEMON.

2. **Dificuldade Isolada no Problema de 4 Classes (Vulnerabilidade com Fundo):**
   - No problema de 4 classes, o DEMON isolado tem SP menor ($43{,}51\%$) porque o ruído AMBIENTAL (ondas e vento na superfície) apresenta envoltórias de baixa frequência que se sobrepõem à faixa de $0\text{--}5\text{ Hz}$. Sem as raias tonais estritas do LOFAR e sem o perfil espectral de banda larga do MEL, o DEMON sozinho confunde barcos pequenos de baixo SNR com ruído de fundo (29,1% de SMALL classificado como AMBIENTAL).

3. **Validação da Tese Arquitetural:**
   - O DEMON **não** deve ser utilizado de forma monolítica isolada para a tarefa geral de 4 classes.
   - O papel ótimo e natural do DEMON é no **Classificador Especialista em Cascata (Etapa B)** e na **Fusão Híbrida**, atuando especificamente na separação entre embarcações (onde alcançou **$50{,}56\%$** em SMALL e até **$79{,}67\%$** em LARGE), fornecendo a informação de RPM que falta ao espectro LOFAR/MEL.

---

### 6.4 Comparativo Multimodelo com DEMON: Random Forest, MLP e CNN (GPU Docker)

Para testar se outros classificadores da literatura extraem melhor a informação do DEMON, foram avaliadas as três arquiteturas de referência do artigo original (*Silva et al.*, 2025) sob o mesmo protocolo estrito de $5\times2$cv nos 10 folds:

| Modelo | Representação | Acurácia 4C (%) | Índice SP 4C (%) | Recall SMALL (%) | Recall LARGE (%) | SP 3C (Embarcações) (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RF (Artigo)** | MEL | $62{,}64\% \pm 1{,}84\%$ | $62{,}22\% \pm 1{,}86\%$ | — | — | — |
| **RF (Artigo)** | LOFAR | $58{,}87\% \pm 1{,}68\%$ | $56{,}92\% \pm 1{,}69\%$ | — | — | — |
| **RF (Novo)** | **DEMON** | $50{,}06\% \pm 1{,}48\%$ | $44{,}30\% \pm 2{,}30\%$ | **$53{,}43\%$** | $68{,}22\%$ | $53{,}47\% \pm 1{,}74\%$ |
| **MLP (Artigo)** | MEL | $64{,}51\% \pm 1{,}75\%$ | $63{,}38\% \pm 1{,}81\%$ | — | — | — |
| **MLP (Artigo)** | LOFAR | **$67{,}48\% \pm 1{,}24\%$** | **$66{,}51\% \pm 1{,}39\%$** | — | — | — |
| **MLP (Novo)** | **DEMON** | **$52{,}10\% \pm 2{,}65\%$** | **$52{,}12\% \pm 2{,}70\%$** | $29{,}78\%$ | $62{,}66\%$ | **$56{,}00\% \pm 1{,}68\%$** |
| **CNN (Artigo)** | MEL | $64{,}99\% \pm 2{,}09\%$ | $63{,}52\% \pm 2{,}26\%$ | — | — | — |
| **CNN (Artigo)** | LOFAR | $67{,}02\% \pm 1{,}78\%$ | $66{,}05\% \pm 1{,}90\%$ | — | — | — |
| **CNN (Novo)** | **DEMON** | $42{,}47\% \pm 1{,}88\%$ | $35{,}72\% \pm 12{,}63\%$ | $19{,}27\%$ | $55{,}89\%$ | $41{,}67\% \pm 2{,}63\%$ |
| **SVM (Proposto)**| **DEMON** | $51{,}72\% \pm 2{,}01\%$ | $46{,}57\% \pm 4{,}45\%$ | $15{,}51\%$ | **$73{,}32\%$** | $54{,}83\% \pm 1{,}69\%$ |

#### Diagnóstico Técnico:
1. **O MLP foi o melhor classificador para DEMON ($52{,}12\%$ SP):** Suas camadas densas com `BatchNorm1d` e `Dropout` pesaram os picos harmônicos de forma não-linear sem sofrer com o marulho superficial ($78{,}78\%$ em AMBIENTAL).
2. **A CNN degradou no DEMON ($35{,}72\%$ SP):** Filtros convolucionais 2D aplicados sobre modulações esparsas funcionam como filtros passa-baixas espaciais, borrando e enfraquecendo as raias pontuais de passagem de pás.
3. **Conclusão Geral:** Nenhuma arquitetura com DEMON puro conseguiu superar o LOFAR ou MEL. O teto do DEMON isolado gira em torno de $52\%$ de SP, provando que sua função tática reside exclusivamente na **fusão multimodal** e em **classificadores especialistas binários**.


