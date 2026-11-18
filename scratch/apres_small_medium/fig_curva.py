import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
H="reports/small_medium/"; OUT="presentation/20261116-eda-iara-dados/figuras/"
COL={"SMALL":"#2a78d6","MEDIUM":"#eb6834","LARGE":"#1baf7a"}
sm=pd.read_csv(H+"curva_navios_sm.csv"); lg=pd.read_csv(H+"curva_navios_large.csv")
fig,ax=plt.subplots(figsize=(6.6,3.5))
for cn,d,nc in [("SMALL",sm,"n_small"),("MEDIUM",sm,"n_medium"),("LARGE",lg,"n_large")]:
    g=d.groupby("frac").agg(x=(nc,"mean"),y=(f"R_{cn}","mean"),e=(f"R_{cn}","sem"))
    print(cn, g.round(1).to_dict("list"))
    ax.errorbar(g.x,g.y,yerr=g.e,color=COL[cn],lw=2,marker="o",ms=6,mec="white",mew=1.5,capsize=3)
    ax.annotate(cn,(g.x.iloc[-1],g.y.iloc[-1]),xytext=(7,0),textcoords="offset points",color=COL[cn],fontweight="bold",va="center",fontsize=9)
ax.axhline(25,color="#999",lw=1,ls="--"); ax.text(120,21.5,"acaso (25%)",fontsize=8,color="#666")
ax.axvspan(40,55,color="#888",alpha=.08); ax.text(41,76,"mesmo nº\nde navios",fontsize=7.5,color="#52514e")
ax.set_xscale("log"); ax.set_xticks([10,20,40,50,100,200]); ax.set_xticklabels(["10","20","40","50","100","200"])
ax.set_xlim(8,330); ax.set_ylim(0,85)
ax.set_xlabel("navios da classe no treino (eixo log)"); ax.set_ylabel("recall em navios novos (%)")
ax.grid(alpha=.25,lw=.6)
fig.tight_layout(); fig.savefig(OUT+"curva_navios.png",dpi=200); print("ok")
