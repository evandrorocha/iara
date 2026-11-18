import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0,"scratch/apres_small_medium"); sys.path.insert(0,"scratch")
from aug_ship_split_modelos import augment, OFF
from eval_ship_split_5x2cv import CACHES, FOLDS_CSV, fold_files, matrices, metadata, fit_predict
folds=pd.read_csv(FOLDS_CSV,dtype={"Ship ID":str}); meta=metadata(); ship_of={f:(s if s!="-" else f"bg_{f}") for f,s in meta["Ship ID"].items()}
cache=pickle.load(open(CACHES["lofar"][0],"rb")); mix=pickle.load(open("data/iara_processed/cache_lofar_mixbg.pkl","rb"))
tr,te=fold_files(folds,"split_navio",0,"trn"),fold_files(folds,"split_navio",0,"test")
for v in ["dup","stretch","mask","mixnavio","ruido","bg"]:
    c2,t2=augment(cache,tr,v,0,ship_of,mix); ex=[f for f in t2 if f>=OFF]
    d=np.mean([np.corrcoef(c2[f].ravel(),cache[f-OFF].ravel())[0,1] for f in ex[:50]])
    print(f"{v}: {len(ex)} gravações extras de {len(tr)}; correlação média com o original {d:.3f}; rótulos ok {all(t2[f]==tr[f-OFF] for f in ex)}")
Xtr,ytr,_=matrices(cache,tr); Xte,yte,fte=matrices(cache,te); pred,_=fit_predict("rf",Xtr,ytr,Xte)
orig=pd.read_csv("results/trainings/multirep/split_navio_rf_lofar_4c/eval/fold_0/split_navio_rf_lofar_4c_multiclass_test.csv")
print("RF base fold 0 igual ao NB29:", np.array_equal(pred, orig.Prediction.values))
