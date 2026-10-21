# Auditoria independente: projeto IARA (Claude, 19/10/2026)

**Escopo**
- Verificar se os itens de `reports/auditoria_completa_iara.md` (auditoria do Gemini) foram sanados.
- Checar se as próprias afirmações do Gemini estão corretas.
- Fazer uma auditoria nova dos 25 notebooks, dos scripts e dos relatórios.

**Método**
- Nenhum número foi aceito por estar escrito em notebook.
- As métricas foram recalculadas a partir dos dados brutos:
  - CSVs por janela em `results/trainings/tests/*/eval/fold_k/`, com voto majoritário por arquivo;
  - pickles em `data/iara_processed/`.
- SP = √(média(E)·média_geom(E)), igual ao `src/iara/ml/metrics.py`.
- Tabelas do artigo conferidas no PDF (Tabelas 7–10).
- Significância testada com o mesmo teste do artigo (F-test 5×2cv de Alpaydin).
- Nenhum arquivo do projeto foi alterado, exceto a criação deste relatório.

---

## 0. Resumo executivo

| # | Achado | Gravidade |
|---|---|---|
| 1 | A **cascata** reportada (NB6: SP 68,54%; apresentação: 69,48%, "+3,23 p.p. sobre o artigo") ainda usa o especialista SMALL×LARGE **com vazamento**. O código foi corrigido, mas **o modelo nunca foi retreinado** (pastas de junho). Estimativa sem vazamento: **≈66,1%** (cascata no nível de arquivo) e **≈67%** (melhor cascata por janela). Isso fica dentro do desvio do MLP-LOFAR do artigo (66,51±1,39). | CRÍTICA |
| 2 | O **Gating** (NB24) foi corrigido para `fold ^ 1` (sem vazamento). Porém a configuração do roteador é a **melhor ou quase a melhor de 18 variantes** (máxima exata em MLP e CNN), o que indica seleção no teste. Além disso, **nenhum** modelo com gate difere significativamente da melhor entrada isolada (F-test 5×2cv, p = 0,36–0,71). | ALTA |
| 3 | Os "oráculos" e o "Super-Oráculo 97,66%" (NB10–22, 24, 25) **não demonstram complementaridade**. 4 preditores *aleatórios* já dão SP ≈ 67%, e 16 aleatórios dão **≈ 99%**. Os oráculos observados ficam **abaixo** do esperado sob independência, ou seja, os erros são positivamente correlacionados, o oposto da tese dos notebooks. | CRÍTICA (interpretação) |
| 4 | A "**Votação simples ≥2/4**" conta empates 2×2 como acerto (é um oráculo parcial). Com voto real, a vantagem some ou inverte (ex.: CNN SP 57–60 vs. 63,8 do melhor modelo isolado). | ALTA |
| 5 | A **CNN LOFAR** usada nas análises tem SP **36,2±13,3%** (artigo: 66,05). O modelo colapsou (folds com recall 0 em uma classe), mas é usado sem aviso nos NBs 20–22, 24 e 25 (ex.: "Gated CNN +27,73% vs LOFAR"). | ALTA |
| 6 | O **isolamento por navio (`Ship ID`)** é falso: **63,5%** das gravações de navio no teste têm o mesmo navio no treino. O protocolo é o mesmo do artigo (que isola só por gravação), então **o erro está apenas em afirmar** o isolamento. Foi corrigido nos NBs 1–23, mas **continua no NB25**, em `reports/implementacao_gating_frontend.md` e em `reports/evandro.rocha.relatorio.tex:186`. | ALTA |
| 7 | Parte das "correções" do Gemini **introduziu erros novos**: recalls do Híbrido no NB6; "cada arquivo é avaliado 5 vezes" (é ~3,74); tabela de baselines "sem vazamento" do Gemini com valores errados. | ALTA |
| 8 | Os notebooks agora estão **executados**, mas **continuam hardcoded** (exceto NB24 e parte do NB25). A execução apenas ecoa os literais. | MÉDIA |

**Conclusão para o objetivo do trabalho (subir o SP via recall de SMALL e MEDIUM).** Com os dados atuais, nenhuma das três estratégias (cascata por janela, cascata por arquivo, gate de entrada) tem ganho **estatisticamente demonstrado** sobre o melhor baseline do artigo (MLP LOFAR 66,51±1,39). A cascata é a mais promissora: o estágio 1 vai de 63,84 para ≈66,1 sem vazamento, com SMALL de 38→54%. Mas o número de 68,5–69,5% não pode ser publicado.

---

## 1. Status dos itens do relatório do Gemini

Legenda:
- ✅ corrigido;
- ⚠️ parcial;
- ❌ não corrigido;
- 🟥 **o Gemini estava errado**.

### 1.1 Categorias transversais

| Item Gemini | Status | Evidência |
|---|---|---|
| #1 Vazamento no Gating (9 folds) | ✅ | `scratch/test_competence_gating_all_models.py`, `_svm.py` e `run_heterogeneous_gating_experiment.py` usam `trn_fold = fold ^ 1` com `assert` de interseção vazia. O NB24 lê os pickles e reproduz os valores (recalculei). **Porém** `reports/implementacao_gating_frontend.md` ainda mostra os números com vazamento (69,82 / 70,68), e o NB25 cél. 12 ainda cita "70,68% com o H-MoE". |
| #2 Vazamento dos especialistas SVM | ❌ **(só o código mudou)** | O código de `svm_binary_specialist*.py` foi corrigido, mas **nenhum modelo foi retreinado**: todas as pastas `*_binary_*` são de 05–06/jun. `eval_cascade_by_ship.py` ainda aponta para `..._small_large_pca8_...`. Medi: **1925/3920 = 49,1%** dos arquivos SMALL/LARGE de teste do 4-classes estão no **treino** do especialista SL. O NB6 agora diz "avaliado de forma estrita", o que é pior que antes. |
| #2 (sub) "o mesmo ocorre no SMALL×MEDIUM" (arquivos 27, 31, 32, 33, 38) | 🟥 | Sobreposição medida: **0/3270 (0%)**. Esses arquivos estão no treino do especialista, mas **também** no treino do 4-classes. Motivo: o `StratifiedKFold` embaralha classe por classe com o mesmo RNG. Ao filtrar {0,1} (as duas primeiras classes) as partições ficam idênticas por coincidência; ao filtrar {0,2}, o LARGE "herda" o estado aleatório do MEDIUM e desalinha. Os especialistas PyTorch SL também têm 0%. |
| #3 Falsa alegação de `Ship ID` | ⚠️ | Corrigido nos NBs 1–23. Persiste em: NB25 cél. 0; `reports/implementacao_gating_frontend.md` L6; `reports/evandro.rocha.relatorio.tex` L186 ("o mesmo navio nunca aparece simultaneamente em treino e teste"); docstrings de `scratch/save_svm_all_representations.py`, `generate_rf_/generate_cnn_predictions_and_analysis.py` e `eval_bigru_*`. Números: 1445 gravações de navio no UO, 586 navios, 1034 gravações de navios com mais de uma gravação; **63,5%** das gravações de navio no teste compartilham `Ship ID` com o treino. |
| #4 Acurácia balanceada × micro | ⚠️ | NB1 e NB4 ✅. NB2, NB7 e NB8 continuam misturando na mesma coluna/curva; no NB2 isso cria uma falsa queda da CNN. **NB6 ❌**: as linhas 0–2 da coluna "Acurácia Balanceada" são acurácia micro (ex.: LOFAR m=6000 = 63,63, mas a média dos recalls é 65,47), e a coluna "Acurácia Micro" das linhas 0 e 2 não corresponde a nenhuma métrica. |
| #5 Hardcoded / não executado | ⚠️ | Todos foram executados sem erro. Os NBs 1–23 seguem com dicts literais. Nos NBs 1–4 e 10–22 os literais conferem com os dados brutos; nos NBs 5–8 e 23 há valores errados (seção 2). |

### 1.2 Itens por notebook

| NB | Item Gemini | Status | Observação |
|---|---|---|---|
| 1 | F1 "Micro" com valores Macro | ✅ | |
| 1 | "Exclusive Ships on Test" | ✅ | |
| 2 | Linhas CNN t≥0,6 e t≥0,7 duplicadas | 🟥 | São iguais de verdade: a CNN tem 7 segmentos por arquivo, 4/7 < 0,6 e 5/7 ≥ 0,7. |
| 2 | "MEDIUM 82,9%" vs. tabela | ✅ | |
| 2 | Método por softmax ≠ Tabela 9 | 🟥 parcial | O método já é concordância de votos por arquivo, como no artigo. Ficou só a palavra "softmax" no texto. |
| 3 | Forest C→A 59,59 | ✅ | |
| 3 | Números locais ≠ Tabela 10 | ✅ | Agora rotulados "Re-execução local". |
| 4 | Mistura BA/micro | ✅ | |
| 5 | m=6000 × m=4000 | ✅ | |
| 5 | 3C copiado do 4C | ⚠️ | "3C" significa coisas diferentes na mesma coluna (ver 2.4). |
| 6 | Recalls "alucinados" do Híbrido | ❌ **+ 🟥** | O NB adotou o vetor que o Gemini disse ser o real, [57,75, 56,11, 67,28, 79,59], e **ele também está errado**. Recalculado de `svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5`: **≈50,3 / 64,8 / 71,4 / 76,5** (SP 65,13). A média do vetor do NB (65,18) nem bate com a BA da mesma linha (65,77). |
| 6 | `n_mels=64` | ✅ | |
| 7 | 50,6/31,9 × matriz | ✅ | |
| 7 / 11 | "17.060 predições" | ✅ | |
| 8 | MEL 0–11 kHz / DEMON 0–50 Hz | ✅ | Erro novo na contagem de bins: são 4 descartados, não 3. |
| 8 | MLP DEMON NB7 ≠ NB8 | ⚠️ | Explicado (25 vs. 35 épocas), mas há um 3º "MLP DEMON" no pickle de complementaridade (SP 53,23). |
| 8 | 3C "Baseline Artigo" inventado | ✅ | Agora vem do pickle, mas o método é otimista: descarta arquivos com alvo **ou predição** = BG. |
| 9 | hop 5,12 s não documentado | ✅ | Não diz explicitamente que metade do áudio é descartada. |
| 9 | Picos 139,1/231,2 Hz | ✅ | |
| 9 | VMD sobre envelope DEMON | ⚠️ | O texto da cél. 14 ("Modo 2 = 9,98 Hz, Modo 3 = 19,99 Hz") contradiz a saída (4,94 / 9,98 / 20,00 Hz). |
| 10 | 7,75% → 7,36% | ✅ | |
| 10/12/13 | Troca silenciosa de modelo SVM | ⚠️ | Declarada, mas há **3** SVMs HHT/DEMON distintos com o mesmo rótulo (m=2000 no pickle do gate, m=4000 nos NBs 10–13, m=6000 nos NBs 9/11). |
| 10/12/13 | "Áudios únicos" | ⚠️ | O texto mudou, mas a correção introduziu "cada arquivo é avaliado 5 vezes" (NBs 10–16). O real é **3,74 em média** (1 a 5). Títulos de gráfico ainda dizem "Áudios Únicos". |
| 10/12/13 | ≥2/4 com empates | ⚠️ | Declarado, mas as conclusões "supera o melhor modelo" continuam (ver 2.2). |
| 10, 14–16 | Caminhos de imagem | ✅ | NBs 17–19 ainda sem fallback `../`. |
| 14–16 | Assimetria MLP | ⚠️ | Declarado "25 épocas", mas o script usa **30** (`generate_mlp_predictions_and_analysis.py` L87). Sem validação e com desempate diferente, não declarados. |
| 15/16 | LaTeX `\\ge` | ✅ | NBs 17–22 têm `\\%`/`\\ge` quebrados (60+ ocorrências) e TAB literal em `\text` (NB18/19). |
| 17/20 | MLP 41,74% → 49,89% | ✅ | |
| 17/20 | SMALL ≤15 m | ✅ | |
| 18 | Gráficos faltando | ⚠️ | Linha "Campeão" adicionada; os gráficos continuam ausentes. |
| 21 | 6 células trocadas | ✅ | Os valores "reais" do Gemini estavam certos. |
| 22 | 8 células trocadas | ✅ | Idem. |
| 23 | Duas arquiteturas BiGRU | ✅ no essencial | Ainda não declarados: `fc_hidden` 32 vs. 64, 2 BatchNorm, `drop_last`, pesos de classe diferentes. |
| 24 | §5 duplicada; "economia de 70%" | ✅ | |
| 25 | "164 gravações" | ❌ | 164 instâncias = **98 gravações únicas, de 75 navios**. |
| — | Tabela "sem vazamento" do Gemini (seção 3.2) | 🟥 | Baselines errados: RF LOFAR é **63,12** (não 60,72); SVM LOFAR é **63,83** (não 63,12); CNN LOFAR é **36,24** local (66,05 é do artigo, e foi misturado). |

---

## 2. Achados novos (não apontados pelo Gemini)

### 2.1 Cascatas (estratégias 1 e 2)

Reconstruí a cascata no nível de arquivo (NB6) a partir dos CSVs por janela:

| Versão | SP | BA | Recall S/M/L/B |
|---|---|---|---|
| Estágio 1 (LOFAR SVM m=6000) | 63,84 ± 2,01 | 65,48 | 38,2 / 60,5 / 76,3 / 86,9 |
| Cascata **com vazamento** (= NB6) | **68,54** ± 1,79 | 69,16 | 58,7 / 60,5 / 70,5 / 86,9 |
| Cascata **sem vazamento**\* | **66,13** ± 1,96 | 66,85 | 53,5 / 60,5 / 66,5 / 86,9 |

\* Para cada arquivo que o especialista viu no treino, usei a predição do especialista do fold irmão da mesma repetição, treinado na outra metade e que nunca viu o arquivo.

Nas cascatas **por janela** (melhor resultado da apresentação: 69,48%), comparei arquivos que o especialista SL viu no treino com os que não viu:

| Classe | Arquivo visto no treino do SL | Arquivo não visto |
|---|---|---|
| SMALL | 75,8% | **64,3%** |
| LARGE | 70,7% | **63,0%** |

Com os recalls dos arquivos não vistos, o SP fica em torno de **67%**. Todas as cascatas acima de 66,5% nos resultados usam o especialista SL com vazamento. A melhor cascata **sem vazamento** com 10 folds completos é a `..._small_medium_pca64_..._smallalso_large_svm_nystroem_6000_lofar...`, com **SP 65,35**.

**Onde isso aparece:**
- NB6 (68,54);
- `presentation/apresentacao-resultados.tex` L904/976/995 (69,48; "+3,23 p.p."; diz que o baseline do artigo é "66,25%", mas o **artigo diz 66,51**);
- `reports/comparacao_modelos.md` L74–76 / 249–256 (68,05 / 67,59 / 67,58).

### 2.2 Gating (NB24)

- **Sensibilidade a hiperparâmetros.** Rodei 18 variantes do roteador (`max_depth` ∈ {3, 5, None} × `min_samples_leaf` ∈ {1, 5} × peso do prior ∈ {0, 0,1, 0,3}):

| Família | Configuração reportada (d3, l5, w0,1) | Faixa nas 18 variantes | Mais próxima da pré-auditoria (d5, l1, w0) |
|---|---|---|---|
| RF | 64,56 | 62,97–64,74 | 63,97 |
| MLP | 65,88 (= máximo) | 64,01–65,88 | 65,39 |
| SVM | 64,42 | 60,37–64,45 | 62,85 |
| CNN | 63,97 (= máximo) | 60,01–63,97 | 62,50 |

  Os ganhos alegados (+0,17 a +1,44 p.p.) estão **dentro da variação causada pela escolha de hiperparâmetros**. Essa escolha não tem conjunto de validação próprio.
- **F-test 5×2cv (gate vs. melhor entrada isolada):**

| Família | F | p |
|---|---|---|
| RF | 1,16 | 0,46 |
| MLP | 0,69 | 0,71 |
| SVM | 1,43 | 0,37 |
| CNN | 0,77 | 0,66 |

  Expressões como "Ganho expressivo" (NB24 cél. 12) não são sustentadas.
- A linha CNN compara com o **LOFAR colapsado** (36,24) para anunciar "+27,73%".
- NB24 cél. 0 e 12: "particiona as **1.278** gravações". O UO tem **1.706** gravações; 1.278 = 2 × 639 (os conjuntos de teste de um par de folds; o restante é validação).
- Os especialistas não são homogêneos dentro da família:
  - MLP: MEL/LOFAR vêm do pipeline oficial (com early stopping); HHT/DEMON vêm de script ad hoc (30 épocas, sem validação).
  - CNN: MEL é oficial; LOFAR/HHT/DEMON vêm de uma CNN ad hoc.
  - SVM: HHT/DEMON com m=2000.
  - RF: 100 árvores sem `max_depth` (o artigo usa 50/30 e 200/20).

- **Adendo (20/10) — "MEL" e "LOFAR" dos especialistas RF e CNN não são as representações do artigo.**
  - `generate_rf_predictions_and_analysis.py` usa `cache_fusion_mel_demon.pkl['mel']` e `cache_lofar.pkl`. São médias sobre **5,12 s** (`build_lofar_cache.py`): o arquivo 1 tem 55 frames, contra 551 nas features oficiais de 0,512 s.
  - `generate_cnn_predictions_and_analysis.py` usa o mesmo `cache_lofar.pkl`.
  - Portanto "RF MEL", "RF LOFAR" e "CNN LOFAR" no gate e nos NBs 17–22 têm resolução temporal 10× menor que a do artigo.
  - Na CNN, uma imagem de 32 frames corresponde a 164 s. A maioria das gravações (2–5 min) gera uma única imagem, completada com *wrap padding*. Isso explica bem o colapso da CNN-LOFAR (SP 36%).
  - Os experimentos exportados para `results/trainings/multirep/` levam o sufixo `mel5s` / `lofar5s`.

### 2.3 Oráculos e votação (NBs 10–22, 24, 25)

| Comparação | Valor |
|---|---|
| Oráculo de 4 preditores uniformemente aleatórios | SP ≈ 67%; recall por classe ≈ 68% |
| Oráculo de 16 preditores aleatórios | SP ≈ 99% (o "Super-Oráculo" observado é 97,66%) |

A frase "Teto físico, não estatístico" (NB25 cél. 12) é inválida.

Oráculo observado vs. esperado sob independência (1 − ∏(1−rᵢ)):

| Caso | Observado | Esperado |
|---|---|---|
| SVM SMALL | 84,0 | 89,9 |
| SVM MEDIUM | 83,7 | 96,1 |
| MLP LARGE | 88,8 | 99,1 |

Isso mostra erros **correlacionados**, o que contradiz "os erros NÃO são nos mesmos áudios" (NB10) e "NUNCA concordam no erro" (NB10 cél. 10). Em 499 dos 877 erros do MEL em SMALL, o LOFAR erra para a mesma classe.

Voto real (pluralidade com desempate aleatório) vs. "≥2/4" reportado:

| Notebook | Reportado | Voto real | Melhor modelo isolado |
|---|---|---|---|
| NB12 | 64,63 | 59,4 | 60,4 |
| NB14 | 49,89 | 43,8 | 46,35 |
| NB17 | 65,62 | 59,7 | — |

Em SP global, o voto real da **CNN piora**: 57–60 vs. 63,8.

"Resgates exclusivos" são instâncias de fold e instáveis. Exemplo: "117 navios" (NB16) = 69 gravações, das quais só **9** são exclusivas em todas as aparições. Outras contagens: NB17 "372 Áudios Únicos" = 208 gravações; NB19 "961 navios" = 302 gravações.

### 2.4 Números errados ou inconsistentes (NBs 5–8, 11, 23, 25)

- **NB5/7/8:** recall SMALL do SVM LOFAR m=6000 = **42,58**; o correto é **38,15** (provavelmente um valor 3C rotulado como 4C).
- **NB8:** linha "SVM m=6000 MEL" é o modelo m=4000.
- **NB5–7:** valores de referência hardcoded em `scratch/eval_demon_svm_5x2cv.py` L304–305 divergem dos CSVs (ex.: MEL SMALL 50,90 vs. 50,73; LOFAR-PCA64 SMALL 47,81 vs. 47,64). O NB1 usa os corretos, então o projeto contradiz a si mesmo.
- **NB5/7/8:** "3C" mistura modelos retreinados sem BG com modelos 4C filtrados por alvo **ou predição** ≠ BG (otimista). A conclusão do NB7 de que o DEMON "supera com folga o LOFAR" compara 3C retreinado contra 4C.
- **NB5 cél. 5:** picos DEMON do LARGE (1,80 Hz / 105 RPM; BPF 5,5 Hz) não existem em `demon_features/4.pkl`. O argmax é 0,39 Hz e os picos são 18,55 / 7,13 / 21,0 Hz. Além disso, 1,80 × 60 = 108 RPM, não 105.
- **NB6:** o Híbrido usa **C=0,5** com normalização (não declarado). A variante C=2,0 dá 64,66, o que indica escolha pelo teste.
- **NB6:**
  - "0–500 Hz" para 250 bins LOFAR: o correto é 0–1953 Hz (Δf = 7,81 Hz).
  - "TPSW zerou o fundo oceânico": falso (4,4% de zeros, mesmo perfil do SMALL).
  - "σ = 1,00%" do MEL: o SP_std da tabela é 1,12.
- **NB9/10:** MEL "0–11 kHz" e "0–24 kHz"; o correto é 0–8 kHz após reamostragem para 16 kHz.
- **NB11 cél. 3:** a linha LOFAR mistura recalls do m=4000-PCA64 com SP/BA do m=6000.
- **NB11 cél. 6:** compara RF-HHT local (58,14) com RF-LOFAR do **artigo** (56,92). O RF-LOFAR local é **63,12**, então o HHT não supera o LOFAR.
- **NB23:**
  - "L=15 frames ≈ 7,5 s": os frames têm passo de **5,12 s** (`demon_features/1.pkl`: 5,12; 10,24; …), então 15 frames ≈ **77 s**.
  - Recalls de baseline errados: SVM MEL MED/LARGE é 60,4/69,3, não 54,1/75,4; CRNN LOFAR é 37,6/68,4/62,5.
  - Não há validação nem early stopping (épocas fixas, 214 arquivos de validação descartados).
- **NB25 cél. 6:** números dos arquétipos contradizem as próprias saídas:

| Arquétipo | Texto | Saída |
|---|---|---|
| 1 | 39 | 35 |
| 2 | 29 | 28 |
| 3 | 24 | 26 |

  As faixas de meta-features também não conferem: `high_low_ratio` de 5 a 19 vale para só 3/35. "Bow thrusters", "radares da Marinha" e "AHTS 8.000 HP" não têm base nos dados.

### 2.5 Superalegações sem teste estatístico

O artigo compara modelos pelo F-test 5×2cv; os notebooks não o usam. Resultados:

| Alegação | Onde | p | Observação |
|---|---|---|---|
| "PCA essencial no LOFAR" | NB1/4 | 0,14 | O SVM LOFAR m=6000 **sem** PCA tem BA maior. |
| "PCA degrada o MEL" | NB1/4 | 0,40 | Em m=4000, o PCA64 é ligeiramente **melhor**. |
| "SVM superou a MLP" no CPA | NB3 | 0,59 | Em SP a MLP é maior. O NB3 omite que o SVM LOFAR tem a maior queda A→C (−10,97 p.p.). |
| "Fusão equivale ou supera o LOFAR" | NB8 | — | Compara fusão local com a Tabela 7. O RF LOFAR local (63,12) supera a RF fusão (62,05). A MLP fusão usa rede maior ([64,32]) e janela de 5,12 s. |
| "Metade da variância" | NB1/4 | — | É metade do **desvio**; a variância é ~1/3. |

### 2.6 O que foi confirmado como correto

- NB1–4: tabelas inteiras, incluindo as 14 execuções do NB4 e a Tabela 7 do artigo.
- NB3: tabela e valores citados da Tabela 10.
- NB12–22: aritmética das análises de complementaridade, que bate com os pickles.
- NB21/22: valores corrigidos da Seção 8.
- NB24: valores atuais, que batem com os pickles.
- NB25: aritmética 6390/164/97,43%.
- Folds idênticos aos oficiais em todos os pickles e no BiGRU.
- Especialistas SMALL×MEDIUM sem vazamento.
- DEMON: 48 kHz → 50 Hz, 253 bins de 0,39 a 25 Hz.

---

## 3. Recomendações (em ordem)

1. **Retreinar** os especialistas SMALL×LARGE com o `svm_binary_specialist_large.py` já corrigido (`--override`), rodar de novo `eval_cascade_by_ship.py` e as cascatas por janela. Só então atualizar NB6, `comparacao_modelos.md` e a apresentação.
2. **Fixar os hiperparâmetros do roteador** sem olhar o teste: validação interna no `fold ^ 1` ou nested CV. Reportar o F-test 5×2cv contra a melhor entrada isolada e contra o MLP-LOFAR do artigo.
3. **Remover ou reescrever as interpretações de oráculo e "≥2/4".** Incluir o baseline aleatório (1−∏(1−rᵢ)) e reportar voto real.
4. **Corrigir ou excluir a CNN LOFAR/HHT/DEMON ad hoc**, ou declarar que colapsou.
5. **Remover as alegações de `Ship ID`** (NB25, relatório `.tex`, `implementacao_gating_frontend.md`). Opcionalmente, rodar um experimento com isolamento real por navio, que tende a baixar todos os números.
6. **Fazer os notebooks lerem os resultados de CSV/pickle** em vez de dicts literais. Isso elimina toda a classe de erros de transcrição encontrada aqui.
7. **Padronizar** a métrica de acurácia (BA, como no artigo), o desvio-padrão (ddof) e a definição de "SVM proposto" (m e PCA) em todos os notebooks.

Scripts de verificação (fora do repositório): `verify_pickles.py`, `verify_cascade.py`, `gate_sensitivity.py`, entre outros, no diretório de rascunho da sessão.
