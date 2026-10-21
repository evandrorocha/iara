"""
Versão versionável dos resultados de results/trainings/ (que não vai para o git: 88 GB, quase tudo modelos).

    results/predicoes/<caminho relativo a results/trainings>.csv.gz

Cada CSV de results/trainings/ (exceto pastas model/ e os *_all.csv, que são a união de trn+val+test) é copiado em gzip para results/predicoes/, com o mesmo
caminho relativo. CSVs de predição (colunas File, Target, Prediction) são reduzidos a essas três colunas:
todas as métricas do projeto (iara_results, notebooks, apresentação, relatórios) usam só elas; as colunas
Prediction_k (probabilidades/escores por classe) ficam só na cópia local completa. Outros CSVs (métricas,
splits) e os JSON de configuração são copiados inteiros.

Uso (na raiz do repositório):
    python scripts/predicoes.py exportar     # results/trainings -> results/predicoes (após novos experimentos)
    python scripts/predicoes.py restaurar    # results/predicoes -> results/trainings (clone novo)
    python scripts/predicoes.py verificar    # confere que a cópia reproduz File/Target/Prediction

`notebooks/iara_results.py` chama `restaurar()` sozinho quando results/trainings/ não existe.
"""
import gzip
import os
import shutil
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "results", "trainings")
DST = os.path.join(ROOT, "results", "predicoes")
SLIM = ["File", "Target", "Prediction"]


def _origens():
    for base, dirs, files in os.walk(SRC):
        dirs[:] = sorted(d for d in dirs if d != "model")
        for f in sorted(files):
            if f.endswith("_all.csv"):   # união de trn+val+test: redundante, nenhuma análise lê
                continue
            if f.endswith(".csv") or (f.endswith(".json") and not f.startswith(".")):
                yield os.path.join(base, f)


def _gz_write_bytes(data: bytes, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(data)


def exportar(verbose=True):
    n = n_slim = 0
    for src in _origens():
        rel = os.path.relpath(src, SRC)
        dst = os.path.join(DST, rel + ".gz")
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            continue
        if src.endswith(".csv"):
            head = pd.read_csv(src, nrows=0).columns.tolist()
            if all(c in head for c in SLIM):
                df = pd.read_csv(src, usecols=SLIM)[SLIM]
                _gz_write_bytes(df.to_csv(index=False).encode("utf-8"), dst)
                n_slim += 1
            else:
                with open(src, "rb") as f:
                    _gz_write_bytes(f.read(), dst)
        else:
            with open(src, "rb") as f:
                _gz_write_bytes(f.read(), dst)
        n += 1
        if verbose and n % 500 == 0:
            print(f"  {n} arquivos...", flush=True)
    print(f"exportados {n} arquivos ({n_slim} CSVs de predição reduzidos a {SLIM}) em {DST}")


def restaurar(verbose=True):
    """Recria results/trainings/**.csv|.json a partir de results/predicoes/ (não sobrescreve o que existe)."""
    n = 0
    for base, _, files in os.walk(DST):
        for f in files:
            if not f.endswith(".gz"):
                continue
            src = os.path.join(base, f)
            dst = os.path.join(SRC, os.path.relpath(src, DST)[:-3])
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with gzip.open(src, "rb") as gz, open(dst, "wb") as out:
                shutil.copyfileobj(gz, out)
            n += 1
    if verbose:
        print(f"restaurados {n} arquivos em {SRC}")
    return n


def verificar():
    erros = 0
    for src in _origens():
        if not src.endswith(".csv"):
            continue
        dst = os.path.join(DST, os.path.relpath(src, SRC) + ".gz")
        if not os.path.exists(dst):
            print("FALTA", dst)
            erros += 1
            continue
        a = pd.read_csv(src)
        b = pd.read_csv(dst)
        cols = SLIM if all(c in a.columns for c in SLIM) else list(a.columns)
        if not a[cols].equals(b[cols]):
            print("DIFERENTE", dst)
            erros += 1
    print("verificação:", "OK" if erros == 0 else f"{erros} problemas")
    return erros


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "exportar":
        exportar()
    elif cmd == "restaurar":
        restaurar()
    elif cmd == "verificar":
        sys.exit(1 if verificar() else 0)
    else:
        print(__doc__)
        sys.exit(2)
