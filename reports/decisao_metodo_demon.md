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
   - **Etapa A:** Classificação isolada com SVM Nyström (medir capacidade de separação $5\times2$cv).
   - **Etapa B:** Inclusão de DEMON no classificador especialista em cascata (substituindo ou complementando LOFAR-PCA8 com DEMON).
