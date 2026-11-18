import sys, numpy as np, pandas as pd, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"scratch"); from eval_ship_split_5x2cv import metadata
OUT="presentation/20261116-eda-iara-dados/figuras/"
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#52514e","axes.labelcolor":"#0b0b0b"})
m=metadata(); m["Length"]=pd.to_numeric(m.Length,errors="coerce")
COL={"SMALL":"#2a78d6","MEDIUM":"#eb6834","LARGE":"#1baf7a"}
# 1) curva de concentração
fig,ax=plt.subplots(figsize=(6.2,3.4))
for c,cn in [(0,"SMALL"),(1,"MEDIUM"),(2,"LARGE")]:
    n=m[m.Target==c].groupby("Ship ID").size().sort_values(ascending=False).values
    x=np.arange(1,len(n)+1); y=np.cumsum(n)/n.sum()*100
    ax.plot(x,y,lw=2,color=COL[cn]); i={"SMALL":20,"MEDIUM":30,"LARGE":150}[cn]; ax.annotate(f"{cn} ({len(n)} navios)",(x[i],y[i]),xytext={"SMALL":(-95,8),"MEDIUM":(8,-14),"LARGE":(8,-10)}[cn],textcoords="offset points",color=COL[cn],fontsize=9,fontweight="bold")
for c,cn,k in [(0,"SMALL",5),(1,"MEDIUM",2)]:
    n=m[m.Target==c].groupby("Ship ID").size().sort_values(ascending=False).values
    v=n[:k].sum()/n.sum()*100; ax.plot([k],[v],"o",ms=7,color=COL[cn],mec="white",mew=2)
    ax.annotate(f"{k} navios = {v:.0f}%",(k,v),xytext=(10,4 if cn=="SMALL" else -12),textcoords="offset points",fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("nº de navios (do mais gravado ao menos gravado; escala logarítmica)"); ax.set_ylabel("% acumulado das gravações")
TK = [1, 2, 5, 10, 20, 50, 100, 200, 417]   # rótulos em números comuns (em vez de 10^0, 10^1, 10^2)
ax.set_xticks(TK); ax.set_xticklabels([str(t) for t in TK]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlim(0.9, 480)
ax.grid(alpha=.25,lw=.6); ax.set_ylim(0,102)
fig.tight_layout(); fig.savefig(OUT+"concentracao_navios.png",dpi=200)
# 2) composição por tipo
def tipo(r):
    t=r["DETAILED TYPE"]
    if r.Target==0: return {"Pilot Vessel":"Prático (Pilot)","Tug":"Rebocador (Tug)","Fishing Vessel":"Pesca"}.get(t,"Outros")
    return {"Motor Hopper":"Motor Hopper (HANG BO)","Offshore Supply Ship":"Offshore Supply (OSS)"}.get(t,"Outros supply/AHTS")
fig,axes=plt.subplots(2,1,figsize=(6.4,2.9))
pal=["#2a78d6","#eb6834","#1baf7a","#b0afa8"]
for ax,(c,cn) in zip(axes,[(0,"SMALL"),(1,"MEDIUM")]):
    d=m[m.Target==c].copy(); d["t"]=d.apply(tipo,axis=1)
    g=d.groupby("t").agg(g=("Name","size"),n=("Ship ID","nunique")).sort_values("g",ascending=False)
    g=pd.concat([g.drop(index=[i for i in g.index if i.startswith("Outros")]),g.loc[[i for i in g.index if i.startswith("Outros")]]])
    left=0
    for i,(t,r) in enumerate(g.iterrows()):
        w=r.g/len(d)*100; ax.barh(0,w-0.4,left=left,color=pal[i],height=.6)
        if w>25: ax.text(left+w/2,0,f"{t}\n{r.g} grav. / {r.n} navios",ha="center",va="center",fontsize=7.5,color="white")
        elif w>12: ax.text(left+w/2,0,f"{t.split(' (')[0]}\n{r.g} grav. / {r.n} navios",ha="center",va="center",fontsize=6.5,color="white")
        else: ax.text(left+w/2,0,f"{t}\n{r.g} /\n{r.n}",ha="center",va="center",fontsize=6,color="white" if i<3 else "#0b0b0b")
        left+=w
    ax.set_xlim(0,100); ax.set_yticks([0]); ax.set_yticklabels([cn]); ax.set_xticks([]); ax.spines[:].set_visible(False)
fig.tight_layout(); fig.savefig(OUT+"composicao_tipos.png",dpi=200)
print("ok")
