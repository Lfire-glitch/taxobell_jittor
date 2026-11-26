"""模型产出 → 评测桥：投影头前向 → rank_anchors → Node/Edge/Ancestor F1 与 MR/MRR"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
from src import ops_numpy as O  # noqa: E402
from src import projection  # noqa: E402
from src import synth  # noqa: E402

NPZ=os.path.join(ROOT,"logs","parity_input.npz")
OUT=os.path.join(ROOT,"logs","eval_pipeline.json")
INIT_SOURCE="projection.init_weights()，初始权重未训练"


def load_eval_bridge():
    path=os.path.join(ROOT,"scripts","eval_bridge.py")
    spec=importlib.util.spec_from_file_location("eval_bridge",path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_rows():
    with np.load(NPZ,allow_pickle=True) as z:
        n={k:z[k].astype(np.float32) for k in ("n_c","n_p","n_n")}
        rows=[str(s).split("|") for s in z["row_ids"]]
    return n,rows


def candidates(t,n,rows,w):
    mu={}
    for tag,key in (("c","n_c"),("p","n_p"),("n","n_n")):
        mu[tag]=projection.forward_numpy(w,n[key])
    out={}
    for q in t.queries:
        idx=[i for i,(c,p,x) in enumerate(rows) if c==q]
        if not idx:
            continue
        ids=[t.gold_parent[q]]+list(t.negatives[q])
        m=np.stack([mu["p"][0][idx[0]]]+[mu["n"][0][i] for i in idx])
        o=np.stack([mu["p"][1][idx[0]]]+[mu["n"][1][i] for i in idx])
        out[q]=dict(ids=ids,mu=m.astype(np.float32),o=o.astype(np.float32),
                    mu_q=mu["c"][0][idx[0]],o_q=mu["c"][1][idx[0]])
    return out


def rank_all(cands,mode):
    out={}
    for q,c in cands.items():
        s=O.rank_anchors(c["mu_q"][None],c["o_q"][None],c["mu"],c["o"],mode)[0]
        out[q]=[c["ids"][i] for i in np.argsort(-s)]
    return out


def scenario(ev,t,ranked):
    gold=[tuple(e) for e in t.edges]
    pm={c:p for c,p in t.edges}
    pm.update({q:r[0] for q,r in ranked.items()})
    pe=[(c,p) for c,p in pm.items()]
    m=ev.taxobell_metrics(ranked,t.gold_parent)
    return dict(Node_F1=ev.node_f1(pe,gold)[2],
                Edge_F1=ev.edge_f1(pe,gold)[2],
                Ancestor_F1_macro=ev.ancestor_f1(pe,gold,"macro"),
                Ancestor_F1_micro=ev.ancestor_f1(pe,gold,"micro")[2],
                **m)


def load_weights(path):
    if not path:
        return projection.init_weights(),INIT_SOURCE
    with np.load(path) as z:
        w={k:z[k].astype(np.float32) for k in projection.WEIGHT_KEYS}
    return w,f"train.py --save-weights {os.path.basename(path)}，WSL 训练后权重"


def mode_separability():
    q=np.zeros((1,4),np.float32)
    qo=np.ones((1,4),np.float32)
    mu=np.zeros((2,4),np.float32)
    o=np.array([[2.0]*4,[0.5]*4],np.float32)
    bc=O.rank_anchors(q,qo,mu,o,"BC")[0]
    kl=O.rank_anchors(q,qo,mu,o,"KL")[0]
    bg,kg=float(abs(bc[0]-bc[1])),float(abs(kl[0]-kl[1]))
    return dict(bc_gap=bg,kl_gap=kg,separable=bool(bg<1e-5 and kg>1.0))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights",default="")
    ap.add_argument("--tag",default="")
    a=ap.parse_args()
    out=os.path.join(ROOT,"logs",f"eval_pipeline_{a.tag}.json") if a.tag else OUT
    ev=load_eval_bridge()
    t=synth.build()
    n,rows=load_rows()
    w,source=load_weights(a.weights)
    cands=candidates(t,n,rows,w)
    perfect={q:[t.gold_parent[q]]+list(t.negatives[q]) for q in cands}
    broken=dict(perfect)
    for q in t.queries[:len(t.queries)//2]:
        if q in broken:
            broken[q]=[t.negatives[q][0],t.gold_parent[q]]+list(t.negatives[q][1:])
    cases={"perfect":perfect,"broken":broken,
           "model_BC":rank_all(cands,"BC"),"model_KL":rank_all(cands,"KL")}
    res={k:scenario(ev,t,r) for k,r in cases.items()}
    sep=mode_separability()
    dis=sum(1 for q in cases["model_BC"]
            if not np.array_equal(np.asarray(cases["model_BC"][q]),
                                  np.asarray(cases["model_KL"][q])))
    print("== 模型产出 → 评测桥 ==")
    print(f"  source: {source}")
    for k,v in res.items():
        print(f"  {k:9s} MR={v['MR']:.3f} MRR={v['MRR']:.3f} "
              f"EdgeF1={v['Edge_F1']:.4f} AncF1(macro)={v['Ancestor_F1_macro']:.4f}")
    print(f"  BC/KL 协议可分辨={sep['separable']} bc_gap={sep['bc_gap']:.2e} kl_gap={sep['kl_gap']:.3f}")
    print(f"  模型产出 BC/KL 排序不一致 {dis}/{len(cases['model_BC'])}")
    p,b=res["perfect"],res["broken"]
    ok=(p["MRR"]>=b["MRR"] and p["Ancestor_F1_macro"]>b["Ancestor_F1_macro"]
        and abs(p["Ancestor_F1_macro"]-1.0)<1e-9
        and sep["separable"]
        and all(np.isfinite(x) for v in res.values() for x in v.values()))
    print(f"  [{'PASS' if ok else 'FAIL'}] MRR 与 Ancestor F1 同向，完美预测 Ancestor F1=1")
    with open(out,"w",encoding="utf-8") as f:
        json.dump(dict(source=source,weight_fingerprint=projection.weight_fingerprint(w),
                       results=res,mode_separability=sep,
                       model_mode_disagreements=dis,ok=ok,
                       note="perfect/broken 为评测桥自检基线（非模型产出）；model_* 来自 rank_anchors；Wu&P 未纳入"),
                  f,indent=2,ensure_ascii=False)
    print(f"  -> {out}")
    return 0 if ok else 1


if __name__=="__main__":
    sys.exit(main())
