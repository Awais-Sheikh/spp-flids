import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
NAVY="#365F91"; LIGHT="#D9EAF7"; ORG="#E07B39"; LORG="#FDEBDD"; GRN="#5B9A68"; LGRN="#E8F3E8"; GREY="#8C8C8C"; INK="#1b2a41"
def box(ax,x,y,w,h,t,fc=LIGHT,ec=NAVY,fs=7.4,lw=1.1,bold=False,ls="-"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.0",fc=fc,ec=ec,lw=lw,ls=ls))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=INK,fontweight="bold" if bold else "normal")
def arr(ax,x1,y1,x2,y2,c=NAVY,ls="-",lw=1.1,ms=9,cs="arc3"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=ms,lw=lw,color=c,ls=ls,connectionstyle=cs))

# ---------------- Figure 1: system architecture ----------------
fig,ax=plt.subplots(figsize=(7.2,4.5)); ax.set_xlim(0,100); ax.set_ylim(0,66); ax.axis("off")
xs=[2,27,52,77]; names=["Client 1","Client 2","···","Client K"]
for x,n in zip(xs,names):
    if n=="···":
        ax.text(x+10.5,38,"···",ha="center",va="center",fontsize=16,color=GREY); continue
    ax.add_patch(FancyBboxPatch((x,18),21,46,boxstyle="round,pad=0.3,rounding_size=1.2",fc="white",ec=NAVY,lw=1.2,ls="--"))
    ax.text(x+10.5,61,n,ha="center",va="center",fontsize=8,fontweight="bold",color=INK)
    box(ax,x+1.5,48,18,8,"Private local traffic\nstays on the client")
    box(ax,x+1.5,35,18,9,"Local IDS (MLP)\nproximal term μ",fc=LGRN,ec=GRN,fs=7.2)
    box(ax,x+1.5,21.5,18,9,"Clip to C, add\nnoise σC",fc=LORG,ec=ORG,fs=7.2)
    arr(ax,x+10.5,48,x+10.5,44.4); arr(ax,x+10.5,35,x+10.5,30.4)
box(ax,12,4,52,9,"Secure aggregation: pairwise masks cancel in the sum\n(simulated in this study)",fc=LORG,ec=ORG,fs=7.2)
for x in [xs[0],xs[1],xs[3]]:
    arr(ax,x+10.5,21.3,x+10.5+(42-(x+10.5))*0.45,13.4,c=ORG)
box(ax,70,4,27,9,"Aggregation server:\nuniform mean → θᵗ⁺¹",fs=7.2)
arr(ax,64.3,8.5,69.7,8.5)
ax.annotate("",xy=(99.3,60),xytext=(99.3,13.6),arrowprops=dict(arrowstyle="-|>",color=GREY,lw=1.1,ls="--",connectionstyle="arc3,rad=0"))
ax.text(100.8,36,"broadcast θᵗ⁺¹ to all clients",rotation=90,ha="center",va="center",fontsize=7,color=GREY)
plt.savefig("fig_arch.png",dpi=200,bbox_inches="tight"); plt.close()

# ---------------- Figure 2: workflow ----------------
fig,ax=plt.subplots(figsize=(7.2,3.5)); ax.set_xlim(0,100); ax.set_ylim(0,48); ax.axis("off")
top=[("1  Server sends\nglobal model θᵗ",LIGHT,NAVY),("2  Local training\n(E = 5 epochs,\nproximal term μ)",LGRN,GRN),("3  Update\nΔθₖ = θₖ − θᵗ",LGRN,GRN),("4  Clip to C and\nadd noise σC",LORG,ORG)]
bot=[("5  Secure aggregation\n(simulated masking)",LORG,ORG),("6  Server: uniform\nmean, θᵗ⁺¹ = θᵗ + Δ̄",LIGHT,NAVY),("7  Validation macro-F1\nevery round",LIGHT,NAVY),("8  After T = 20 rounds:\none test evaluation",LIGHT,NAVY)]
for i,(t,fc,ec) in enumerate(top): box(ax,2+i*24.5,30,21,12,t,fc=fc,ec=ec,fs=7.4)
for i in range(3): arr(ax,23.4+i*24.5,36,26.4+i*24.5,36)
for i,(t,fc,ec) in enumerate(bot): box(ax,2+i*24.5,6,21,12,t,fc=fc,ec=ec,fs=7.4)
for i in range(3): arr(ax,23.4+i*24.5,12,26.4+i*24.5,12)
arr(ax,86,29.6,86,18.6)
ax.text(87.5,24,"to server",fontsize=7,color=GREY,va="center")
arr(ax,2,18.4,2,29.6,c=GREY,ls="--")
ax.text(3,24,"repeat for t = 0 … T−1",fontsize=7,color=GREY,va="center")
ax.text(50,0.8,"Executed setting: K = 10 clients, full participation, T = 20 rounds, 3 seeds; raw records never leave a client.",ha="center",fontsize=7.2,color="#444")
plt.savefig("fig_workflow.png",dpi=200,bbox_inches="tight"); plt.close()

# ---------------- Figure 3: threat model ----------------
fig,ax=plt.subplots(figsize=(7.2,3.8)); ax.set_xlim(0,100); ax.set_ylim(-5,52); ax.axis("off")
hdr=[("Adversary",1,22),("Target",26,22),("Control in the pipeline",51,24),("Status in this study",78,21)]
for t,x,w in hdr: ax.text(x+w/2,49,t,ha="center",va="center",fontsize=8,fontweight="bold",color=NAVY)
rows=[("External attacker\nmalicious IoT traffic","Network traffic\nat the client","Local IDS\n(detection)",("Evaluated:\n15-class detection",GRN,LGRN)),
      ("Curious server\nupdate inference","Individual client\nupdates","Secure aggregation\n+ clipping and DP noise",("Partly: ε accounted,\nSecAgg simulated",ORG,LORG)),
      ("Malicious client\npoisoning, backdoor","Global model\n(the aggregate)","Update clipping\n(+ noise, proximal term)",("Partly: label flipping\nonly; no backdoor",ORG,LORG))]
for r,(a,b,c,(s,ec,fc)) in enumerate(rows):
    y=33-r*15.5
    box(ax,1,y,22,11,a,fc="#F3F3F3",ec=GREY,fs=7.3); box(ax,26,y,22,11,b,fs=7.3); box(ax,51,y,24,11,c,fs=7.3); box(ax,78,y,21,11,s,fc=fc,ec=ec,fs=7.1)
    arr(ax,23.4,y+5.5,25.6,y+5.5,c=GREY); arr(ax,48.4,y+5.5,50.6,y+5.5,c=GREY); arr(ax,75.4,y+5.5,77.6,y+5.5,c=GREY)
ax.text(50,-3,"Not covered: server-side robust aggregation, collusion with the server, dropout, model extraction.",ha="center",fontsize=7.2,color="#444")
plt.savefig("fig_threat.png",dpi=200,bbox_inches="tight"); plt.close()
from PIL import Image
ims=[Image.open(f) for f in ["fig_arch.png","fig_workflow.png","fig_threat.png"]]
w=760; ims=[im.resize((w,int(im.height*w/im.width))) for im in ims]
S=Image.new("RGB",(w,sum(i.height for i in ims)),"white"); y=0
for im in ims: S.paste(im,(0,y)); y+=im.height
S.save("sheet_arch.png"); print(S.size)
