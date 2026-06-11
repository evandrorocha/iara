import pickle
import os
import glob
import numpy as np
import pandas as pd

# Load a few pickle files to compute variance and scale gamma for MEL
data_dir = r"c:\Users\ev_ro\git\IARA\data_processed\log_melgram_0abb0b4e967cfa3ba82fee9233b681f2"
pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))[:20]

dfs = []
for file_path in pkl_files:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        df = data.get('df', None)
        if df is not None:
            dfs.append(df.values)

if dfs:
    X = np.vstack(dfs)
    print(f"Loaded MEL features with shape: {X.shape}")
    n_features = X.shape[1]
    var = X.var()
    mean = X.mean()
    gamma_scale = 1.0 / (n_features * var)
    gamma_auto = 1.0 / n_features
    print(f"--- MEL FEATURES ---")
    print(f"Number of features (d): {n_features}")
    print(f"Mean: {mean:.6f}")
    print(f"Variance (Var(X)): {var:.6f}")
    print(f"Calculated gamma='scale' (1 / (d * Var(X))): {gamma_scale:.6f}")
    print(f"Calculated gamma='auto' (1 / d): {gamma_auto:.6f}")

# Let's also do it for LOFAR
lofar_dir = r"c:\Users\ev_ro\git\IARA\data_processed\lofar_9deb0c89447d0c92ed5a436303929ede"
lofar_files = glob.glob(os.path.join(lofar_dir, "*.pkl"))[:20]

dfs_lofar = []
for file_path in lofar_files:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        df = data.get('df', None)
        if df is not None:
            dfs_lofar.append(df.values)

if dfs_lofar:
    X_lofar = np.vstack(dfs_lofar)
    print(f"\nLoaded LOFAR features with shape: {X_lofar.shape}")
    n_features = X_lofar.shape[1]
    var = X_lofar.var()
    mean = X_lofar.mean()
    gamma_scale = 1.0 / (n_features * var)
    gamma_auto = 1.0 / n_features
    print(f"--- LOFAR FEATURES ---")
    print(f"Number of features (d): {n_features}")
    print(f"Mean: {mean:.6f}")
    print(f"Variance (Var(X)): {var:.6f}")
    print(f"Calculated gamma='scale' (1 / (d * Var(X))): {gamma_scale:.6f}")
    print(f"Calculated gamma='auto' (1 / d): {gamma_auto:.6f}")
