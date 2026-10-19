import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if "NotoSansCJK-Regular" in f or "NotoSansCJK-Bold" in f: font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family":"Noto Sans CJK JP","axes.unicode_minus":False,"font.size":17,
 "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#9aa3ad","axes.labelcolor":"#33404d",
 "xtick.color":"#33404d","ytick.color":"#33404d","axes.grid":True,"grid.color":"#e3e7eb","savefig.bbox":"tight",
 "figure.facecolor":"none","axes.facecolor":"none","svg.fonttype":"none"})
BLUE,ORANGE,GREEN,GOLD,PINK,INK="#1f5fae","#e8622c","#16a37b","#e0a100","#c4508a","#16202b"

# 1. grokking 개념 곡선 (E11 측정 시점 기준: t_fit≈200, seen 50%≈4,800, t_gen≈6,000)
s=np.logspace(0,4.3,600)
tr=1/(1+np.exp(-(np.log10(s)-np.log10(130))*9))
seen=0.12*(1/(1+np.exp(-(np.log10(s)-2.4)*8)))+0.88/(1+np.exp(-(s-4800)/450))
te=0.03+0.97/(1+np.exp(-(s-5900)/380))
fig,ax=plt.subplots(figsize=(9,4.3))
ax.axvspan(200,5900,color=GOLD,alpha=.10,lw=0)
ax.plot(s,tr,color=BLUE,lw=3.5,label="train 정확도")
ax.plot(s,te,color=ORANGE,lw=3.5,label="test 정확도")
ax.set_xscale("log"); ax.set_ylim(-.03,1.08); ax.set_xlim(1,2e4)
ax.set_xlabel("학습 step (로그)"); ax.set_ylabel("완전 일치 정확도")
ax.annotate("외우기 완료\n(약 200 step)",xy=(200,1),xytext=(12,.62),fontsize=14,color=BLUE,
            arrowprops=dict(arrowstyle="->",color=BLUE,lw=1.6))
ax.annotate("뒤늦은 일반화\n(약 6,000 step)",xy=(5900,.5),xytext=(1500,.25),fontsize=14,color=ORANGE,ha="right",
            arrowprops=dict(arrowstyle="->",color=ORANGE,lw=1.6))
ax.text(1100,1.035,"지연 = t_gen − t_fit",ha="center",fontsize=14,color="#8a6a00",weight="bold")
ax.legend(loc="center left",bbox_to_anchor=(0,.42),frameon=False)
fig.savefig("fig_grok.svg")

# 2. 대칭으로 묶으면 (E6, 1,024개 지연 중앙값)
fig,ax=plt.subplots(figsize=(6.2,4.6))
lab=["S₄\n(24)","6","Z₅\n(5)","2","1\n(없음)"]; v=[6700,4400,400,6900,2500]
ax.bar(range(5),v,color=[PINK,GOLD,BLUE,GREEN,"#7d8a96"],width=.62)
ax.set_xticks(range(5)); ax.set_xticklabels(lab); ax.set_yscale("log"); ax.set_ylim(150,2e4)
ax.set_xlabel("등변군 크기 (대칭이 큰 순)"); ax.set_ylabel("일반화 지연 (step)")
for i,x in enumerate(v): ax.text(i,x*1.15,f"{x:,}",ha="center",fontsize=13)
ax.set_title("대칭 크기와 지연: 순서가 없다",fontsize=16,color=INK,weight="bold",loc="left")
ax.grid(axis="x",visible=False)
fig.savefig("fig_sym.svg")

# 3. 칸당 관찰 수 (E6·E7 + E12 사전 예측)
pts=[]
def add(Ns,ds,p,cells,g):
    for N,d in zip(Ns,ds):
        if d: pts.append((N*p/cells,d,g))
add([768,896,1024,1152,1280],[4900,2500,400,200,100],5,1275,"ab")
add([896,1024,1152,1280],[6400,2400,1050,250],4,1275,"ab")
add([896,1024,1152,1280],[9300,4400,2300,700],4,1275,"ab")
add([1024,1152,1280],[7600,3900,2100],3,1275,"ab")
add([1024,1152,1280],[24550,9900,5850],5,2500,"a3b")
add([1024,1152,1280],[21100,9600,5800],5,2500,"a3b")
e12={2.6:[11050,15700,10150,4450,6000,5500],3.0:[4850,4650,5100,2100,3300,3150],3.4:[2900,2300,3100,550,2400,1900],2.2:[13450,11900,15150]}
fig,ax=plt.subplots(figsize=(11.5,4.4))
x=np.linspace(1.9,4.2,50); ax.plot(x,10**(5.78-0.754*x),"--",color="#7d8a96",lw=2,label="E12 전에 적어 둔 예측선")
for g,c,l in [("ab",BLUE,"a+b (배선 4종)"),("a3b",ORANGE,"a+3b (교환법칙 없음)")]:
    q=[p for p in pts if p[2]==g]; ax.scatter([p[0] for p in q],[p[1] for p in q],s=70,color=c,label=l,zorder=3)
xs=[k for k,v in e12.items() for _ in v]; ys=[d for v in e12.values() for d in v]
ax.scatter(xs,ys,s=80,facecolors="none",edgecolors=GREEN,lw=2,label="E12 새 조건 (예측 검증)",zorder=3)
ax.set_yscale("log"); ax.set_xlabel("칸당 평균 관찰 수  =  N × (예제당 서로 다른 쌍) / 연산 표 칸 수")
ax.set_ylabel("일반화 지연 (step)"); ax.legend(frameon=False,fontsize=13,loc="upper right")
ax.set_title("하나의 곡선 (상관 −0.93)",fontsize=16,color=INK,weight="bold",loc="left")
fig.savefig("fig_obs.svg")
print(len(pts))
