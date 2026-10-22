"""E39 보충: 외운 오답이 예제 전체(다른 숫자들)에 묶여 있는가.
오답 자리 (행 r, 칸 p)에서 규칙과 무관한 입력 숫자 하나(p, p+1이 아닌 자리)를 다른 값으로 바꾼 뒤에도 외운 오답을 내는지 본다.
흩어진 오답이 예제 단위로 저장됐다면 오답이 사라지고, 칸 단위(cell)로 저장됐다면 그대로 남는다. 2층 attention 분포도 함께.
    python analyze_noise_swap.py results_dir/runs_e39_*/w5_*_s0
"""
import glob, numpy as np, torch, sys
from src.data import compute_targets, load_split
from src.model import build_model
from src.noise import make_noise
from src.rules import Rule
torch.set_grad_enabled(False)
rng=np.random.default_rng(0)
def load(path):
    sd=torch.load(path,weights_only=False,map_location='cpu'); c=sd['config']
    m=build_model('m0',c['n'],c['m'],c['d'],c['layers'],c['heads'],None,c['init_std']).eval(); m.load_state_dict(sd['model']); return m,c
for d in sorted({p for g in sys.argv[1:] for p in glob.glob(g)}):
    ck=sorted(glob.glob(d+'/ckpt_*.pt'))
    if not ck: continue
    tag=d.split('runs_e39_')[-1]
    model,c=load(ck[-1])
    data='data/'+c['data'].split('/')[-1]
    rule=Rule(c['rule_spec'],c['n']); sp=load_split(data); x=sp['train'].astype(np.int64); y=compute_targets(rule,x).astype(np.int64)-1
    n,m=c['n'],c['m']
    if c.get('noise_frac'):
        nz=make_noise(rule,x,y,c['noise_frac'],c['noise_mode'],c['noise_seed']); ynz=nz['y_noisy']; rows,pos=np.nonzero(nz['mask'])
    else:
        ynz=y; rows=rng.choice(len(x),300,replace=False); pos=rng.integers(0,m,300)
    X=torch.as_tensor(x-1)
    # attention of block 1 from query p to inputs
    model.hidden(X,record_attn=True); a=model.blocks[1].attn.last_attn.mean(1)  # B,T,T
    def att(rs,ps):
        q=a[rs,m+ps,:m]  # (K, m)
        self_=q[np.arange(len(rs)),ps]; ref=q[np.arange(len(rs)),(ps+1)%m]
        return float(self_.mean()),float(ref.mean()),float(q.sum(1).mean()-self_.mean()-ref.mean())
    clean=np.nonzero(~(ynz!=y).any(1))[0][:2000]; cp=rng.integers(0,m,len(clean))
    # context swap: change one irrelevant input
    keep_w=keep_t=tot=0
    for r,p in zip(rows,pos):
        for q in range(m):
            if q in (p,(p+1)%m): continue
            row=x[r].copy(); choices=[v for v in range(1,n+1) if v not in row]; row[q]=rng.choice(choices)
            out=model.predict(torch.as_tensor(row[None]-1))[0,p].item()
            keep_w+= out==ynz[r,p]; keep_t+= out==y[r,p]; tot+=1
    orig=(model.predict(X[rows]).numpy()[np.arange(len(rows)),pos]==ynz[rows,pos]).mean()
    print(f"{tag:45s} 원래 출력=오답 {orig:.2f} | 무관한 숫자 1개 바꾼 뒤: 오답 유지 {keep_w/tot:.2f}, 원래 정답 {keep_t/tot:.2f} | 2층 attention(자기/참조/나머지) 오답자리 {att(rows,pos)} 정상 {att(clean,cp)}")
