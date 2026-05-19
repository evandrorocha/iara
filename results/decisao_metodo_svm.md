# Decisão de Método: Por que Nyström + SGD (avaliado by_audio)

## Contexto

O artigo original do IARA (Silva et al., IEEE Access 2025) avaliou três baselines:
**Random Forest**, **MLP** e **CNN**. Nenhum deles incluiu SVM.

O objetivo do trabalho é avaliar SVM neste dataset como contribuição original,
comparando com os modelos já existentes.

---

## Percurso de Decisão

### 1. Por que SVM?

- SVM é coberto na disciplina (Aula 15)
- SVM não foi avaliado no artigo original do IARA → lacuna a preencher
- SVM tem justificativa teórica: **maximização da margem** como proteção contra overfitting
- Os modelos existentes (MLP e CNN) apresentam gap significativo entre treino e teste:
  - MLP: ~67% treino vs ~59% teste (gap de 8%)
  - CNN: ~76% treino vs ~63% teste (gap de 13%)
- Hipótese: o princípio de minimização estrutural de risco da SVM pode reduzir esse gap

---

### 2. Por que não usar SVM RBF direto (sklearn SVC)?

O dataset IARA, quando convertido em janelas espectrais, gera:

```
~853 arquivos de treino × ~586 janelas/arquivo ≈ 500.000 amostras
```

O sklearn SVC com kernel RBF tem complexidade **O(n²) a O(n³)**:

- 500.000² = 250 bilhões de operações
- Tempo estimado: semanas a meses → **inviável**

---

### 3. Por que não usar SVM RBF com vetores médios por arquivo?

Uma alternativa seria calcular a média das janelas de cada arquivo (853 amostras)
e treinar o SVM RBF nessas médias. Isso seria computacionalmente trivial.

**Problema fundamental:** as classes 0 e 1 do IARA são definidas temporalmente:

| Classe | Descrição |
|---|---|
| **0** | Navio próximo **entrando** (aproximando — Doppler crescente) |
| **1** | Navio próximo **saindo** (afastando — Doppler decrescente) |

O mesmo navio, à mesma distância mínima, produz **assinaturas espectrais médias
quase idênticas**. A única distinção entre as classes 0 e 1 é a
**evolução temporal da frequência** ao longo do áudio.

Calcular a média das janelas **apaga exatamente essa informação**, tornando as
classes 0 e 1 praticamente indistinguíveis para o classificador.

```
Classe 0 — entrando:      Classe 1 — saindo:
frequência ↑ com tempo    frequência ↓ com tempo

   ___/                    \___
```

**Média das janelas → vetores quase iguais → SVM não consegue separar as classes.**

Conclusão: abordagem descartada por perda de informação temporal crítica.

---

### 4. Por que não usar LinearSVC direto (sem Nyström)?

LinearSVC escala bem (O(n)), mas usa apenas **kernel linear** — sem não-linearidade.
As features do Mel-espectrograma em 256 dimensões formam espaços altamente não-lineares.
Um kernel linear pode não capturar as fronteiras entre classes acústicas complexas.

Decisão: manter não-linearidade via kernel RBF aproximado.

---

### 5. A solução: Nyström + SGDClassifier

**Nyström** (Williams & Seeger, NeurIPS 2001):
- Aproximação de baixo rank da matriz de kernel RBF
- Seleciona `m` pontos âncora (landmarks) do dataset
- Transforma cada amostra `x` in `z(x) ∈ R^m` via:
  ```
  z(x) = [k(x, âncora₁), k(x, âncora₂), ..., k(x, âncora_m)]
  ```
- A transformação `x → z(x)` é **não-linear** (usa kernel RBF)
- Complexidade: **O(m × n)** → para m=300, n=500K → 150M operações → viável

**SGDClassifier com `loss='hinge'`**:
- Equivalente matemático ao SVM linear
- Resolve o problema de classificação no espaço transformado pelo Nyström
- Linear no espaço `z(x)`, mas **não-linear no espaço original `x`**
- Escala para milhões de amostras

**Pipeline completo:**
```
x (256 mel features)
    ↓ Nyström (kernel RBF aproximado, m=300 componentes)
z(x) ∈ R^300 (espaço transformado, não-linear)
    ↓ SGDClassifier (loss='hinge')
classe predita
```

**Resultado:** equivalente a SVM com kernel RBF, operando janela a janela,
viável computacionalmente em ~50 minutos para 10 folds no Ryzen 9 9950X.

---

### 6. Por que avaliar by_audio?

O framework IARA já implementa a estratégia `by_audio`:
- O modelo prediz a classe de cada **janela** individualmente
- Para cada arquivo de áudio, aplica **majority vote** sobre todas as janelas
- Resulta em uma predição por arquivo

Isso combina o melhor das duas abordagens:
- **Mantém a informação temporal** (cada janela é avaliada individualmente)
- **Reduz ruído** por janelas mal classificadas (votação)

Nos experimentos anteriores com MLP, a estratégia by_audio já mostrou
ganho consistente sobre by_window (~4 pontos percentuais).

---

## Resumo da Decisão

| Opção considerada | Descartada por |
|---|---|
| SVC RBF direto (500K amostras) | Inviável computacionalmente (semanas) |
| SVM RBF por arquivo (média) | Perde informação temporal das classes 0 vs 1 |
| LinearSVC sem Nyström | Kernel linear — perde não-linearidade |
| **Nyström + SGDClassifier** | ✅ Escolhida: viável, não-linear, preserva temporalidade |

**Método escolhido:** `Nystroem(kernel='rbf', n_components=300)` +
`SGDClassifier(loss='hinge')`, avaliado **by_audio** (majority vote por arquivo).

---

## Hipótese Científica

> O classificador SVM com kernel RBF aproximado (Nyström), por maximizar a margem
> entre classes no espaço de features espectrais, apresentará menor overfitting
> (gap treino-teste) do que a MLP, com acurácia competitiva no conjunto de teste
> do dataset IARA.

**Resultado esperado:** gap treino-teste menor que o da MLP (~8%), com
acurácia no teste próxima ou superior a 59.24% (resultado atual da MLP by_audio).

---

*Documento gerado em: 2026-05-18*
*Baseado na conversa de planejamento do trabalho CPE 721.*
