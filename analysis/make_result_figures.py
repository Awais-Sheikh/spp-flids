import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from analysis import *
NAVY="#365F91"; BLUE="#4F81BD"; ORANGE="#E07B39"; GREEN="#5B9A68"; GREY="#8C8C8C"; RED="#C0504D"; LIGHT="#D9EAF7"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold","axes.titlesize":10})
DPI=200
def dots(ax,x,vs,color,off=0.0):
    ax.scatter([x+off+(i-1)*0.045 for i in range(len(vs))],vs,s=14,color=color,edgecolor="white",linewidth=0.5,zorder=4)

# ---- Fig 7: per-class F1 heatmap ----
cols=[("Centralized",CEN),("FedAvg IID",FA['iid']),("FedAvg a=0.5",FA['0.5']),("FedAvg a=0.1",FA['0.1'])]
M=np.array([[np.mean([r['per_class_f1'][c] for r in rs]) for _,rs in cols] for c in range(NCLS)])
order=np.argsort(M[:,0])
fig,ax=plt.subplots(figsize=(6.2,4.8))
im=ax.imshow(M[order],cmap="RdYlGn",vmin=0.2,vmax=1.0,aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([c[0].replace("a=","α=") for c in cols],fontsize=8.5)
ax.set_yticks(range(NCLS)); ax.set_yticklabels([CL[i] for i in order],fontsize=8)
for i in range(NCLS):
    for j in range(4): ax.text(j,i,f"{M[order][i,j]:.2f}",ha="center",va="center",fontsize=7.5,color="black")
ax.spines[:].set_visible(False); ax.tick_params(length=0)
cb=plt.colorbar(im,ax=ax,fraction=0.04,pad=0.02); cb.set_label("Per-class F1 (mean of 3 seeds)",fontsize=8)
ax.set_title("Per-class F1 across training regimes")
plt.savefig("fig_perclass.png",dpi=DPI,bbox_inches="tight"); plt.close()

# ---- Fig 8: heterogeneity ----
fig,ax=plt.subplots(figsize=(6.3,3.5))
xs=np.arange(3); w=0.34; labs=["IID","Dirichlet α = 0.5","Dirichlet α = 0.1"]
for j,(nm,D,col) in enumerate([("FedAvg",FA,BLUE),("FedProx (μ = 0.001)",FP,ORANGE)]):
    m=[ms(D[a])[0] for a in ['iid','0.5','0.1']]; s=[ms(D[a])[1] for a in ['iid','0.5','0.1']]
    ax.bar(xs+(j-0.5)*w,m,w,yerr=s,capsize=3,color=col,label=nm,alpha=0.9,error_kw=dict(lw=1))
    for i,a in enumerate(['iid','0.5','0.1']): dots(ax,xs[i],vals(D[a]),"#222",(j-0.5)*w)
ax.axhline(ms(CEN)[0],ls="--",color=NAVY,lw=1.1,label=f"Centralized ({ms(CEN)[0]:.3f})")
ax.set_xticks(xs); ax.set_xticklabels(labs); ax.set_ylim(0.4,0.85); ax.set_ylabel("Test macro-F1")
ax.set_title("Effect of label skew; bars = mean ± SD, dots = individual seeds")
ax.legend(frameon=False,fontsize=8,loc="upper right")
plt.savefig("fig_hetero.png",dpi=DPI,bbox_inches="tight"); plt.close()

# ---- Fig 9: convergence ----
fig,ax=plt.subplots(figsize=(6.3,3.4))
for a,col,lab in [('iid',BLUE,"IID"),('0.5',GREEN,"α = 0.5"),('0.1',ORANGE,"α = 0.1")]:
    H=np.array([r['val_hist'] for r in FA[a]]); m=H.mean(0); s=H.std(0,ddof=1); x=np.arange(1,H.shape[1]+1)
    ax.plot(x,m,color=col,lw=1.8,label=f"FedAvg, {lab}"); ax.fill_between(x,m-s,m+s,color=col,alpha=0.18)
ax.axhline(ms(CEN,'val_f1')[0],ls="--",color=NAVY,lw=1.1,label="Centralized (validation)")
ax.set_xlabel("Communication round"); ax.set_ylabel("Validation macro-F1"); ax.set_ylim(0.3,0.85); ax.set_xticks([1,5,10,15,20])
ax.set_title("FedAvg convergence (mean ± SD over 3 seeds)"); ax.legend(frameon=False,fontsize=8,loc="lower right"); ax.grid(axis="y",alpha=.25)
plt.savefig("fig_conv.png",dpi=DPI,bbox_inches="tight"); plt.close()

# ---- Fig 10: privacy-utility ----
fig,ax=plt.subplots(figsize=(6.3,3.7))
sg=[0.03,0.05,0.1,0.3]
mm=[ms(DP[s])[0] for s in sg]; ss=[ms(DP[s])[1] for s in sg]
ec=[eps(s,K_,T_) for s in sg]; el=[eps(s,1,T_) for s in sg]
ax.errorbar(ec,mm,yerr=ss,fmt="o-",color=BLUE,capsize=3,lw=1.6,label="ε with secure aggregation (σ√K)")
ax.errorbar(el,mm,yerr=ss,fmt="s--",color=ORANGE,capsize=3,lw=1.4,label="ε without secure aggregation (σ)")
for s_,x,y in zip(sg[:3],ec[:3],mm[:3]): ax.annotate(f"σ={s_}",(x,y),textcoords="offset points",xytext=(-4,9),fontsize=7.5,ha="center",color=BLUE)
ax.scatter([ec[-1]],[mm[-1]],marker="x",s=70,color=RED,zorder=6); ax.annotate("σ=0.3: training diverged\nin 3/3 seeds",(ec[-1],mm[-1]),textcoords="offset points",xytext=(-8,-36),fontsize=7.5,color=RED,ha="left")
ax.axhline(ms(CLIP)[0],ls="--",color=NAVY,lw=1.1,label=f"Clipping only, no noise ({ms(CLIP)[0]:.3f})")
ax.set_xscale("log"); ax.set_xlim(20,25000); ax.set_xlabel("Privacy budget ε  (δ = 1e-5, T = 20 rounds, log scale)"); ax.set_ylabel("Test macro-F1"); ax.set_ylim(0,0.85)
ax.set_title("Privacy–utility trade-off (client-level DP, K = 10, α = 0.5)"); ax.legend(frameon=False,fontsize=7.8,loc="lower right")
plt.savefig("fig_privacy.png",dpi=DPI,bbox_inches="tight"); plt.close()

# ---- Fig 11: malicious ----
fig,axs=plt.subplots(1,2,figsize=(6.6,3.2))
for ax,f,yl,tt in [(axs[0],'test_f1',"Test macro-F1","Detection quality"),(axs[1],'asr',"Attack success rate","Label-flip success")]:
    for meth,col,mk,lab in [('FedAvg',BLUE,'o',"FedAvg"),('SPP-FLIDS',ORANGE,'s',"FedProx + clip + DP")]:
        m=[ms(MAL[(meth,k)],f)[0] for k in [0,1,2]]; s=[ms(MAL[(meth,k)],f)[1] for k in [0,1,2]]
        ax.errorbar([0,10,20],m,yerr=s,fmt=mk+"-",color=col,capsize=3,lw=1.6,label=lab)
        for k,x in zip([0,1,2],[0,10,20]): dots(ax,x,vals(MAL[(meth,k)],f),col,0)
    ax.set_xticks([0,10,20]); ax.set_xlabel("Malicious clients (%)"); ax.set_ylabel(yl); ax.set_title(tt)
axs[0].legend(frameon=False,fontsize=7.8,loc="lower left"); axs[0].set_ylim(0.4,0.85); axs[1].set_ylim(0,0.30)
plt.tight_layout(); plt.savefig("fig_malicious.png",dpi=DPI,bbox_inches="tight"); plt.close()

# ---- Fig 12: ablation ----
fig,ax=plt.subplots(figsize=(6.3,3.4))
names=list(ABL); labs=["Full","− FedProx","− Clipping","− DP noise","− SecAgg"]
cl=[NAVY,BLUE,GREEN,ORANGE,"#9B59B6"]
for i,n in enumerate(names):
    ax.bar(i,ms(ABL[n])[0],yerr=ms(ABL[n])[1],capsize=3,color=cl[i],alpha=0.88,hatch="//" if n=="- SecAgg" else None)
    dots(ax,i,vals(ABL[n]),"#222")
ax.set_xticks(range(5)); ax.set_xticklabels(labs); ax.set_ylim(0.4,0.75); ax.set_ylabel("Test macro-F1")
ax.set_title("Ablation at α = 0.1, σ = 0.05 (mean ± SD, dots = seeds)\nhatched bar: identical aggregate to Full; only ε differs",fontsize=9.5)
plt.savefig("fig_ablation.png",dpi=DPI,bbox_inches="tight"); plt.close()

# ---- Fig 13: confusion matrix (centralized, sum of seeds) ----
cm=np.sum([np.array(r['cm']) for r in CEN],axis=0).astype(float); cmn=cm/cm.sum(1,keepdims=True)
fig,ax=plt.subplots(figsize=(6.3,5.6)); im=ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
ax.set_xticks(range(NCLS)); ax.set_yticks(range(NCLS)); ax.set_xticklabels(CL,rotation=60,ha="right",fontsize=7.5); ax.set_yticklabels(CL,fontsize=7.5)
for i in range(NCLS):
    for j in range(NCLS):
        if cmn[i,j]>=0.03: ax.text(j,i,f"{cmn[i,j]*100:.0f}",ha="center",va="center",fontsize=6.5,color="white" if cmn[i,j]>0.5 else "black")
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title("Row-normalised confusion matrix, centralized model (%)")
plt.colorbar(im,ax=ax,fraction=0.04,pad=0.02)
plt.savefig("fig_confusion.png",dpi=DPI,bbox_inches="tight"); plt.close()
# top confusions
off=cm.copy(); np.fill_diagonal(off,0); idx=np.dstack(np.unravel_index(np.argsort(-off.ravel())[:6],off.shape))[0]
print([(CL[i],CL[j],int(off[i,j]),round(off[i,j]/cm[i].sum()*100,1)) for i,j in idx])
print("ok")
