import sys, numpy as np, pandas as pd, scipy.stats as st
sys.path.insert(0,"notebooks/src"); import iara_results as R
V=sys.argv[1].split(",") if len(sys.argv)>1 else ["base","navio","stretch","mask","mixnavio","ruido","navio-stretch-mask"]
base="augnavio_mlp_lofar_base_4c"; fb=R.fold_metrics(base)
rows=[]
for v in V:
    e=f"augnavio_mlp_lofar_{v}_4c"
    try: fm=R.fold_metrics(e)
    except FileNotFoundError: continue
    r={"variante":v}
    for c in ["SP","ACC","R_SMALL","R_MEDIUM","R_LARGE","R_BACKGROUND"]:
        r[c]=f"{fm[c].mean():.1f}±{fm[c].std(ddof=0):.1f}"
    d=fm.ACC-fb.ACC; r["ΔACC"]=round(d.mean(),2); r["p_t"]=round(st.ttest_rel(fm.ACC,fb.ACC).pvalue,3) if v!="base" else None
    r["ΔSP"]=round((fm.SP-fb.SP).mean(),2); r["ΔSM"]=round(((fm.R_SMALL+fm.R_MEDIUM)-(fb.R_SMALL+fb.R_MEDIUM)).mean()/2,2)
    r["F5x2_ACC_p"]=round(R.f_test_5x2(e,base,metric="ACC")[1],3) if v!="base" else None
    r["melhor_ACC"]=int((d>0).sum())
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
