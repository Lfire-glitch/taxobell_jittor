"""全链路自检：投影头(numpy) → 算子损失 → 排父 → 评测桥，纯 numpy / 标准库"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import ops_numpy as O  # noqa: E402
from src import projection  # noqa: E402
from src import synth  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return bool(cond)


def load_eval_bridge():
    path = os.path.join(ROOT, "scripts", "eval_bridge.py")
    spec = importlib.util.spec_from_file_location("eval_bridge", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    print("== 全链路自检（纯 numpy，无框架）==")
    t = synth.build()
    d = synth.export_triplets(t)
    w = projection.init_weights(k=d["n_c"].shape[1], d=d["mu_c"].shape[1])

    check("权重确定性（同 seed 两次一致）",
          projection.weight_fingerprint(w) == projection.weight_fingerprint(
              projection.init_weights(k=d["n_c"].shape[1], d=d["mu_c"].shape[1])),
          projection.weight_fingerprint(w))
    check("权重全 float32", all(v.dtype == np.float32 for v in w.values()))

    mu_c, o_c = projection.forward_numpy(w, d["n_c"])
    mu_p, o_p = projection.forward_numpy(w, d["n_p"])
    mu_n, o_n = projection.forward_numpy(w, d["n_n"])
    check("投影头输出形状", mu_c.shape == d["mu_c"].shape and o_c.shape == d["o_c"].shape,
          f"mu={mu_c.shape} o={o_c.shape}")
    check("投影头 o > 0", bool(np.all(o_c > 0) and np.all(o_p > 0) and np.all(o_n > 0)))
    check("投影头输出全有限", all(np.all(np.isfinite(x)) for x in
                              (mu_c, o_c, mu_p, o_p, mu_n, o_n)))

    lp = O.loss_paper(mu_c, o_c, mu_p, o_p, mu_n, o_n)
    lo = O.loss_official(mu_c, o_c, mu_p, o_p, mu_n, o_n)
    check("论文损失全有限", all(np.isfinite(v) for v in lp.values()),
          f"total={lp['total']:.6f}")
    check("官方损失全有限", all(np.isfinite(v) for v in lo.values()),
          f"total={lo['total']:.6f}")
    check("两套损失数值不等（口径仍不同）",
          abs(lp["total"] - lo["total"]) > 1e-6,
          f"|Δ|={abs(lp['total']-lo['total']):.4f}")

    # 推理：真父 vs 一个 hard negative，两个排序变体都应把真父排前
    mu_a = np.concatenate([mu_p[:1], mu_n[:1]])
    o_a = np.concatenate([o_p[:1], o_n[:1]])
    for mode in ("BC", "KL"):
        s = O.rank_anchors(mu_c[:1], o_c[:1], mu_a, o_a, mode)
        check(f"{mode} 排序把真父排首位", bool(s[0, 0] > s[0, 1]),
              f"{s[0].round(4).tolist()}")

    ev = load_eval_bridge()
    gold = [tuple(e) for e in t.edges]
    check("评测桥完美预测 Node F1=1", abs(ev.node_f1(gold, gold)[2] - 1.0) < 1e-9)
    check("评测桥完美预测 Ancestor F1(macro)=1",
          abs(ev.ancestor_f1(gold, gold, "macro") - 1.0) < 1e-9)

    # 降级记账：把本轮能给出的真实值落盘，供 report / PPT 引用
    ledger = dict(
        l0=synth.l0_fingerprint(t),
        weight_fingerprint=projection.weight_fingerprint(w),
        n_triplets=int(d["mu_c"].shape[0]),
        dim=int(d["mu_c"].shape[1]),
        k=int(d["n_c"].shape[1]),
        hidden=int(w["W1"].shape[1]),
        loss_paper_total=float(lp["total"]),
        loss_official_total=float(lo["total"]),
        backend="numpy",
        note="投影头前向 + 算子损失 + 排父 + 评测桥；未含框架训练。",
    )
    out = os.path.join(ROOT, "logs", "pipeline_selftest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)
    print(f"  记账已写 {out}")

    print(f"\n== {sum(PASS)}/{len(PASS)} PASS ==")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
