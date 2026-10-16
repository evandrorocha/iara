"""
Build cache_lofar.pkl with temporal alignment to DEMON/MEL (55 windows per audio, 1024-D).
"""

import os
import time
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor

LOFAR_DIR = "data/iara_processed/lofar_c5de9d7767d06192303cece6fb0d8783"
DEMON_DIR = "data/iara_processed/demon_features"
CACHE_LOFAR = "data/iara_processed/cache_lofar.pkl"

def process_file(fid):
    f_lof = os.path.join(LOFAR_DIR, f"{fid}.pkl")
    f_dem = os.path.join(DEMON_DIR, f"{fid}.pkl")
    
    with open(f_lof, "rb") as f:
        d_lofar = pickle.load(f)
        df_lofar = d_lofar["df"].values.astype(np.float32)
        tm_lofar = np.array(d_lofar["times"], dtype=np.float32)
        
    with open(f_dem, "rb") as f:
        d_demon = pickle.load(f)
        tm_demon = np.array(d_demon["times"], dtype=np.float32)
        
    aligned = []
    for t in tm_demon:
        mask = (tm_lofar >= (t - 5.12)) & (tm_lofar <= t)
        if np.any(mask):
            aligned.append(df_lofar[mask].mean(axis=0))
        else:
            aligned.append(df_lofar[np.argmin(np.abs(tm_lofar - t))])
            
    return fid, np.array(aligned, dtype=np.float32)

def main():
    if os.path.exists(CACHE_LOFAR):
        print(f"Cache LOFAR já existe em '{CACHE_LOFAR}'.")
        return
        
    print(f"Construindo cache LOFAR de '{LOFAR_DIR}'...")
    t0 = time.time()
    
    fids = [int(os.path.splitext(f)[0]) for f in os.listdir(LOFAR_DIR) if f.endswith(".pkl")]
    print(f"Total de arquivos a processar: {len(fids)}")
    
    cache = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        results = executor.map(process_file, fids)
        for i, (fid, arr) in enumerate(results):
            cache[fid] = arr
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(fids)} arquivos processados ({time.time()-t0:.1f}s)...")
                
    print(f"Salvando {len(cache)} arquivos em '{CACHE_LOFAR}'...")
    with open(CACHE_LOFAR, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Concluído com sucesso em {time.time()-t0:.1f}s!")

if __name__ == "__main__":
    main()
