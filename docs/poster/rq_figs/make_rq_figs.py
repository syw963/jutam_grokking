import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
import numpy as np
plt.rcParams["font.family"]="Noto Sans CJK KR"
NAVY="#26386f"; BLUE="#3f6cb5"; LBLUE="#dfe8f6"; ORG="#e07b39"; LORG="#fbe3d3"; INK="#1d1d1f"; GRAY="#8a8f99"; LG="#eef0f4"
OUT="docs/poster/rq_figs/"

def tri(ax,x0,y0,k,s,fc=LBLUE,ec="white",lw=1):
    for i in range(k):
        for j in range(i,k):
            ax.add_patch(Rectangle((x0+j*s,y0-(i+1)*s),s,s,fc=fc,ec=ec,lw=lw))

# ---------- RQ1 ----------
fig,ax=plt.subplots(figsize=(8,3.6)); ax.set_xlim(0,8); ax.set_ylim(-0.1,3.6); ax.axis("off")
tri(ax,0.35,3.05,6,0.33)
ax.text(1.34,0.78,"숫자 1~23",ha="center",fontsize=13,color=INK,weight="bold")
ax.text(1.34,0.42,"칸 276개",ha="center",fontsize=12,color=BLUE)
tri(ax,3.0,3.15,13,0.18)
ax.text(4.17,0.78,"숫자 1~101",ha="center",fontsize=13,color=INK,weight="bold")
ax.text(4.17,0.42,"칸 5,151개",ha="center",fontsize=12,color=BLUE)
ax.text(0.35,3.3,"덧셈표의 크기 (a+b = b+a라 절반만 센다, 그림은 축소)",ha="left",fontsize=10,color=GRAY)
# flow
steps=[("① 식 세우기","칸 수 → 필요한 예제 수"),("② 미리 예측","새 조건, 실험 전"),("③ 실험으로 확인","예측과 비교")]
for k,(t,s) in enumerate(steps):
    y=2.85-k*0.98
    ax.add_patch(FancyBboxPatch((6.0,y-0.62),1.9,0.72,boxstyle="round,pad=0.02,rounding_size=0.12",fc=LORG if k==1 else LG,ec="none"))
    ax.text(6.95,y-0.17,t,ha="center",fontsize=12,weight="bold",color=ORG if k==1 else NAVY)
    ax.text(6.95,y-0.47,s,ha="center",fontsize=9.5,color=INK)
    if k<2: ax.annotate("",xy=(6.95,y-0.68),xytext=(6.95,y-0.62+0.0),arrowprops=dict(arrowstyle="-|>",color=GRAY,lw=1.5))
ax.text(1.34,0.05,"필요한 예제 수 ?",ha="center",fontsize=11,color=NAVY,weight="bold")
ax.text(4.17,0.05,"필요한 예제 수 ?",ha="center",fontsize=11,color=NAVY,weight="bold")
fig.savefig(OUT+"rq1.png",dpi=300,bbox_inches="tight",facecolor="white"); fig.savefig(OUT+"rq1.svg",bbox_inches="tight")

# ---------- RQ2 ----------
fig,ax=plt.subplots(figsize=(8,3.6)); ax.set_xlim(0,8); ax.set_ylim(-0.1,3.6); ax.axis("off")
xs=[3,17,8,42,25]; bx=0.62
ax.text(0.2,3.3,"예제 하나",fontsize=11,color=GRAY)
for i,v in enumerate(xs):
    ax.add_patch(FancyBboxPatch((0.2+i*bx,2.5),0.52,0.52,boxstyle="round,pad=0.01,rounding_size=0.08",fc=LG,ec="none"))
    ax.text(0.46+i*bx,2.76,str(v),ha="center",va="center",fontsize=13,weight="bold",color=INK)
pairs=["(3,17)","(17,8)","(8,42)","(42,25)","(25,3)"]
for i,p in enumerate(pairs):
    ax.text(0.2,2.05-i*0.36,"이웃한 두 수 "+p if i==0 else p,fontsize=10.5,color=BLUE,
            transform=ax.transData) if False else None
ax.text(1.75,2.15,"이웃한 두 수의 쌍 5개\n(3,17) (17,8) (8,42)\n(42,25) (25,3)",ha="center",va="top",fontsize=10.5,color=BLUE,linespacing=1.5)
ax.text(1.75,1.15,"→ 덧셈표의 칸 5개를 본다",ha="center",va="top",fontsize=11,color=INK,weight="bold",linespacing=1.4)
ax.annotate("",xy=(4.45,1.95),xytext=(3.75,1.95),arrowprops=dict(arrowstyle="-|>",color=GRAY,lw=2))
# grid with counts
rng=np.random.default_rng(3); k=8; s=0.29; x0=4.75; y0=3.1
for i in range(k):
    for j in range(i,k):
        c=rng.choice([0,0,1,1,1,2,2,3,4])
        fc=["#f4f5f8","#dfe8f6","#a9c0e6","#6f95d2",BLUE][c]
        ax.add_patch(Rectangle((x0+j*s,y0-(i+1)*s),s,s,fc=fc,ec="white",lw=1))
        if c: ax.text(x0+j*s+s/2,y0-(i+0.5)*s,str(c),ha="center",va="center",fontsize=7.5,color="white" if c>=3 else NAVY)
ax.text(x0+k*s/2,3.28,"예제 전체로 본 덧셈표 예시 (칸의 수 = 본 횟수)",ha="center",fontsize=10,color=GRAY)
ax.add_patch(FancyBboxPatch((0.15,-0.05),7.75,0.5,boxstyle="round,pad=0.02,rounding_size=0.1",fc=LORG,ec="none"))
ax.text(4.0,0.2,"칸당 관찰 수 = 예제 수 × 예제당 칸 수 ÷ 표 칸 수",ha="center",va="center",fontsize=10.5,color=ORG,weight="bold")
fig.savefig(OUT+"rq2.png",dpi=300,bbox_inches="tight",facecolor="white"); fig.savefig(OUT+"rq2.svg",bbox_inches="tight")
