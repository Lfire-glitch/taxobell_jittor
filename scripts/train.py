"""L2 训练对齐：Jittor 主体 ↔ PyTorch 对照，两端读同一份 npz 与初始权重"""
import argparse
import json
import os
import sys

import numpy as np

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
from src import projection  # noqa: E402

CFG=json.load(open(os.path.join(ROOT,"config","default.json"),encoding="utf-8"))
TOL=CFG["parity"]["l2_rel_loss_tol"]
GATE_STEPS=CFG["parity"]["l2_steps"]
DEFAULT_NPZ=os.path.join(ROOT,"logs","parity_input.npz")
OUT=os.path.join(ROOT,"logs")
TERMS={"paper":("L_diverge","L_reg","L_clip"),"official":("kl_loss","reg_sum")}


def load_batch(path):
    with np.load(path,allow_pickle=True) as z:
        return {k:z[k].astype(np.float32) for k in ("n_c","n_p","n_n")}


def build(backend,w,lr,batch_np):
    if backend=="torch":
        import torch
        from src import model_pytorch as M
        model=M.TaxoBellProjection(w)
        opt=torch.optim.AdamW(model.parameters(),lr=lr)
        batch={k:torch.tensor(v) for k,v in batch_np.items()}
    elif backend=="jittor":
        import jittor as jt
        from src import model_jittor as M
        model=M.TaxoBellProjection(w)
        model.train()
        opt=jt.optim.AdamW(model.param_list(),lr=lr)
        batch={k:jt.array(v) for k,v in batch_np.items()}
    else:
        raise ValueError(backend)
    return model,opt,batch


def run(backend,steps,lr,loss_name,batch_np,save_weights=""):
    w=projection.init_weights()
    model,opt,batch=build(backend,w,lr,batch_np)
    curve=[model.train_step(batch,opt,loss_name) for _ in range(steps)]
    if save_weights:
        if backend=="jittor":
            wout={k:model.params[k].numpy() for k in projection.WEIGHT_KEYS}
        else:
            wout={k:getattr(model,k).detach().numpy() for k in projection.WEIGHT_KEYS}
        np.savez(save_weights,**{k:np.asarray(v,np.float32) for k,v in wout.items()})
    return dict(backend=backend,steps=steps,lr=lr,loss=loss_name,
                weight_fingerprint=projection.weight_fingerprint(w),
                total=[c["total"] for c in curve],
                coverage={k:int(sum(abs(c.get(k,0.0))>1e-12 for c in curve))
                          for k in TERMS[loss_name]},
                last={k:float(v) for k,v in curve[-1].items()})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--backend",choices=("jittor","torch","both"),default="jittor")
    ap.add_argument("--steps",type=int,default=GATE_STEPS)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--loss",choices=("paper","official"),default="paper")
    ap.add_argument("--npz",default=DEFAULT_NPZ)
    ap.add_argument("--tag",default="")
    ap.add_argument("--save-weights",default="")
    a=ap.parse_args()
    if not os.path.exists(a.npz):
        print(f"缺少 {a.npz}，先跑：python scripts/parity.py --emit-input")
        return 1
    batch_np=load_batch(a.npz)
    if a.backend!="both":
        r=run(a.backend,a.steps,a.lr,a.loss,batch_np,a.save_weights)
        print(f"[{r['backend']}] w_fp={r['weight_fingerprint']} "
              f"loss {r['total'][0]:.6f} -> {r['total'][-1]:.6f}")
        with open(os.path.join(OUT,f"train_{a.backend}.json"),"w",encoding="utf-8") as f:
            json.dump(r,f,indent=2,ensure_ascii=False)
        return 0
    rj=run("jittor",a.steps,a.lr,a.loss,batch_np,a.save_weights)
    rt=run("torch",a.steps,a.lr,a.loss,batch_np)
    rels=[abs(x-y)/max(1.0,abs(x)) for x,y in zip(rj["total"],rt["total"])]
    finite_j=all(np.isfinite(x) for x in rj["total"])
    finite_t=all(np.isfinite(x) for x in rt["total"])
    finite=finite_j and finite_t
    ok=finite and max(rels)<TOL
    gate=a.steps<=GATE_STEPS
    verdict="PASS" if ok else ("FAIL" if gate else "DRIFT")
    cuts=sorted({s for s in (0,1,2,9,49,a.steps-1) if s<a.steps})
    print("== L2 逐 step 对齐（Jittor 主体 / PyTorch 对照）==")
    print(f"  npz={os.path.basename(a.npz)} loss={a.loss} steps={a.steps}")
    print(f"  有限性: jittor={finite_j} torch={finite_t} (任一侧 inf/nan 即 FAIL)")
    print(f"  weight_fingerprint jittor={rj['weight_fingerprint']} "
          f"torch={rt['weight_fingerprint']} "
          f"{'一致' if rj['weight_fingerprint']==rt['weight_fingerprint'] else '不一致'}")
    for s in cuts:
        print(f"  step {s:3d} jittor={rj['total'][s]:.8f} "
              f"torch={rt['total'][s]:.8f} rel={rels[s]:.2e}")
    print(f"  max rel={max(rels):.2e}  结论: {verdict} (tol {TOL:g}, gate={gate})")
    for k,v in rj["coverage"].items():
        print(f"  分项覆盖 {k}: {v}/{a.steps} 步非零")
    name="train_parity.json" if not a.tag else f"train_parity_{a.tag}.json"
    out=dict(jittor=rj,torch=rt,max_rel=max(rels) if finite else None,
             rels=rels,tol=TOL,steps=a.steps,loss=a.loss,npz=a.npz,
             finite=dict(jittor=finite_j,torch=finite_t),
             ok=ok,gate=gate,verdict=verdict,
             note="Jittor 是训练主体，PyTorch 仅作数值对照；有限性是先决判据")
    with open(os.path.join(OUT,name),"w",encoding="utf-8") as f:
        json.dump(out,f,indent=2,ensure_ascii=False)
    return 0 if (ok or not gate) else 1


if __name__=="__main__":
    sys.exit(main())
