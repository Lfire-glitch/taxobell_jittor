"""L3 前向双框架对齐；--emit-input 生成 L0 固定输入"""
import argparse
import json
import os
import sys

import numpy as np

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
from src import synth  # noqa: E402

CFG=json.load(open(os.path.join(ROOT,"config","default.json"),encoding="utf-8"))
TOL=CFG["parity"]["l3_max_abs_delta"]
NPZ=os.path.join(ROOT,"logs","parity_input.npz")
FP=os.path.join(ROOT,"logs","l0_fingerprint.json")


def emit_input():
    t=synth.build()
    fp=synth.l0_fingerprint(t)
    with open(FP,"w",encoding="utf-8") as f:
        json.dump(fp,f,indent=2,ensure_ascii=False)
    d=synth.export_triplets(t,out_path=NPZ)
    print("[L0] fingerprint:",json.dumps(fp,ensure_ascii=False))
    print(f"[L0] triplets={d['mu_c'].shape[0]} dim={d['mu_c'].shape[1]} -> {NPZ}")
    print("[L0] 两端必须校验该 JSON 全字段一致后才进入 L1/L2/L3")


def load_npz():
    with np.load(NPZ,allow_pickle=True) as z:
        return {k:z[k].astype(np.float32) for k in
                ("mu_c","o_c","mu_p","o_p","mu_n","o_n")}


def _scalar(x):
    return float(np.asarray(x).item())


def forward_parity():
    import torch
    import jittor as jt
    from src import ops_torch as T,ops_jittor as J
    d=load_npz()
    t={k:torch.tensor(v) for k,v in d.items()}
    j={k:jt.array(v) for k,v in d.items()}
    lp_t,lo_t=T.loss_paper(**t),T.loss_official(**t)
    lp_j,lo_j=J.loss_paper(**j),J.loss_official(**j)
    print("\n== L3 前向逐项 max|Δ| (float32) ==")
    ok=True
    for name,a,b in [("paper:"+k,lp_t[k],lp_j[k]) for k in lp_t]+\
                    [("official:"+k,lo_t[k],lo_j[k]) for k in lo_t]:
        tv,jv=_scalar(a.detach().numpy()),_scalar(b.numpy())
        diff=abs(tv-jv)
        flag=diff<TOL
        ok&=flag
        print(f"  [{'PASS' if flag else 'FAIL'}] {name:22s} torch={tv:.8f} "
              f"jittor={jv:.8f} |Δ|={diff:.2e}")
    print(f"L3 结论: {'PASS' if ok else 'FAIL'} (tol {TOL:g})")
    return ok


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--emit-input",action="store_true")
    a=ap.parse_args()
    if a.emit_input:
        emit_input()
        return 0
    return 0 if forward_parity() else 1


if __name__=="__main__":
    sys.exit(main())
