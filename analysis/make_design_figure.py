import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
NAVY="#365F91"; LIGHT="#D9EAF7"
fig,ax=plt.subplots(figsize=(7.2,2.9)); ax.set_xlim(0,100); ax.set_ylim(0,40); ax.axis("off")
cols=[("Data distribution",["IID","Dirichlet α = 0.5","Dirichlet α = 0.1","K = 10, seeds 0–2"]),
      ("Federated methods",["Centralized MLP","FedAvg","FedProx μ = 0.001, 0.01","Clipping only"]),
      ("Privacy",["σ = 0.03 … 0.3","C = 2.17","ε with / without SecAgg","δ = 1e-5, T = 20"]),
      ("Stress tests",["Label flipping 0/10/20%","Ablation (α = 0.1)","SecAgg overhead","Per-class analysis"])]
for i,(t,items) in enumerate(cols):
    x=1+i*24.5
    ax.add_patch(FancyBboxPatch((x,3),22.5,34,boxstyle="round,pad=0.3,rounding_size=1.2",fc="white",ec=NAVY,lw=1.2))
    ax.add_patch(FancyBboxPatch((x,30),22.5,7,boxstyle="round,pad=0.3,rounding_size=1.2",fc=LIGHT,ec=NAVY,lw=1.2))
    ax.text(x+11.25,33.5,t,ha="center",va="center",fontsize=8.6,fontweight="bold",color="#1b2a41")
    for k,it in enumerate(items): ax.text(x+11.25,25-k*6.4,it,ha="center",va="center",fontsize=7.4,color="#1b2a41")
plt.savefig("fig_design.png",dpi=200,bbox_inches="tight"); plt.close()
