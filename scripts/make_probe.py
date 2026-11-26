"""生成定向输入：让 L_diverge / L_reg / L_clip 在训练全程非零"""
import os
import sys

import numpy as np

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
from src import ops_numpy as O  # noqa: E402
from src import projection  # noqa: E402

NPZ=os.path.join(ROOT,"logs","parity_input.npz")
OUT=os.path.join(ROOT,"logs","probe_input.npz")
KEYS=("n_c","n_p","n_n")
SCALES=((1.0,1.0,1.0),(6.0,10.0,6.0))
TERMS=("L_diverge","L_reg","L_clip")


def main():
    with np.load(NPZ,allow_pickle=True) as z:
        base={k:z[k].astype(np.float32) for k in KEYS}
    blocks=[{k:base[k]*s[i] for i,k in enumerate(KEYS)} for s in SCALES]
    data={k:np.concatenate([b[k] for b in blocks]).astype(np.float32) for k in KEYS}
    w=projection.init_weights()
    mc,oc=projection.forward_numpy(w,data["n_c"])
    mp,op=projection.forward_numpy(w,data["n_p"])
    mn,on=projection.forward_numpy(w,data["n_n"])
    lp=O.loss_paper(mc,oc,mp,op,mn,on)
    print("== 定向样本：三项损失覆盖 ==")
    for k in TERMS:
        print(f"  {k}={float(lp[k]):.6f} {'非零' if abs(lp[k])>1e-9 else '仍为零'}")
    ok=all(abs(lp[k])>1e-9 for k in TERMS) and all(np.isfinite(v) for v in lp.values())
    if not ok:
        print("  FAIL：三项未全部非零或存在非有限值")
        return 1
    np.savez(OUT,**data)
    print(f"  rows={data['n_c'].shape[0]} -> {OUT}")
    return 0


if __name__=="__main__":
    sys.exit(main())
