import sys, os, glob, pickle, numpy as np, pandas as pd
sys.path.insert(0,"scratch/apres_small_medium"); sys.path.insert(0,"scratch")
import aug_ship_split as A
from eval_ship_split_5x2cv import FOLDS_CSV, fold_files, metadata
import scipy.io.wavfile  # noqa
D="data/iara_processed/lofar_mixbg"; fs=sorted(glob.glob(D+"/*.npy"))
jan={tuple(map(int,os.path.basename(f)[:-4].split("_"))):np.load(f) for f in fs}
# reproduz os sorteios (mesma semente do extrator) só para os arquivos existentes
m=pd.read_csv("src/iara/dataset_info/iara.csv",dtype={"ID":str}); bg_ids=sorted(int(i) for i in m[m.Dataset=="E"].ID)
rows=[]
for (f,j) in jan:
    rng=np.random.default_rng(f*100+j); rows.append({"ID":f,"copia":j,"bg_id":int(rng.choice(bg_ids))})
A._MIX={"janelas":jan,"sorteios":pd.DataFrame(rows)}
folds=pd.read_csv(FOLDS_CSV,dtype={"Ship ID":str}); meta=metadata()
ship_of={f:(s if s!="-" else f"bg_{f}") for f,s in meta["Ship ID"].items()}
cache=pickle.load(open("data/iara_processed/cache_lofar.pkl","rb"))
tr,te=fold_files(folds,"split_navio",0,"trn"),fold_files(folds,"split_navio",0,"test")
print("cópias disponíveis:",len(jan))
fte,yte,pred,_=A.run(cache,tr,te,0,"bg",ship_of,epochs=2)
print("ok", len(pred), np.bincount(pred,minlength=4))
