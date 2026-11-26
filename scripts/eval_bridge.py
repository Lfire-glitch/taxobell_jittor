"""评测协议桥：SPARROW 三 F1 + TaxoBell 排序指标（纯 python / numpy，无 LLM、无框架）"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _parent_map(edges):
    return {c: p for c, p in edges}


def _ancestor_fn(edges):
    pm = _parent_map(edges)

    def anc(n):
        out, cur = set(), pm.get(n)
        while cur is not None and cur not in out:
            out.add(cur)
            cur = pm.get(cur)
        return out
    return anc


def _prf(pred_set, gold_set):
    # 两边都为空（如根节点的祖先集）= 完全一致，不是 F1=0。
    # 不这么处理会把根节点算成 0，凭空拉低 macro 平均（实测 35 节点掉到 0.9714）。
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    inter = len(pred_set & gold_set)
    p = inter / len(pred_set) if pred_set else 0.0
    r = inter / len(gold_set) if gold_set else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def node_f1(pred_edges, gold_edges):
    pn = {x for e in pred_edges for x in e}
    gn = {x for e in gold_edges for x in e}
    return _prf(pn, gn)


def edge_f1(pred_edges, gold_edges):
    return _prf({tuple(e) for e in pred_edges}, {tuple(e) for e in gold_edges})


def ancestor_f1(pred_edges, gold_edges, mode="macro"):
    """SPARROW 的 Ancestor F1：逐节点比较预测树与 gold 树的祖先集合。"""
    pa, ga = _ancestor_fn(pred_edges), _ancestor_fn(gold_edges)
    nodes = {x for e in gold_edges for x in e} | {x for e in pred_edges for x in e}
    if mode == "micro":
        P = set().union(*(pa(n) for n in nodes)) if nodes else set()
        G = set().union(*(ga(n) for n in nodes)) if nodes else set()
        return _prf(P, G)
    fs = [_prf(pa(n), ga(n))[2] for n in nodes]
    return float(np.mean(fs)) if fs else 0.0


def taxobell_metrics(ranked, gold_parent, ks=(1, 5, 10)):
    """ranked: {query: [候选父按分数降序]}；gold_parent: {query: 真父}。

    返回 MR↓ / MRR / R@k / H@k。
    """
    rs, mrr, hits = [], [], {k: [] for k in ks}
    for q, cand in ranked.items():
        g = gold_parent.get(q)
        rank = cand.index(g) + 1 if g in cand else len(cand) + 1
        rs.append(rank)
        mrr.append(1.0 / rank if g in cand else 0.0)
        for k in ks:
            hits[k].append(1.0 if g in cand[:k] else 0.0)
    out = {"MR": float(np.mean(rs)), "MRR": float(np.mean(mrr))}
    for k in ks:
        out[f"R@{k}"] = float(np.mean(hits[k]))
        out[f"H@{k}"] = float(np.mean(hits[k]))
    return out


def wup_edge(pred_parent, true_parent, gold_edges):
    """单条预测父边的 Wu&P 相似度（接口级，需 gold 树）。"""
    pm = _parent_map(gold_edges)

    def chain(n):
        out, cur = [], n
        while cur is not None and cur not in out:
            out.append(cur)
            cur = pm.get(cur)
        return out

    cp, ct = chain(pred_parent), chain(true_parent)
    common = [x for x in cp if x in set(ct)]
    if not common:
        return 0.0
    d_lca = len(cp) - cp.index(common[0])       # 越靠前越浅
    return 2 * d_lca / (len(cp) + len(ct))


def _selftest():
    from src import synth
    t = synth.build()
    gold = [tuple(e) for e in t.edges]

    perfect = [tuple(e) for e in t.edges]
    # 碎片化：把一半 query 挂到 root 的兄弟（祖先集合被截断）
    broken = [list(e) for e in t.edges]
    broken_pm = {c: p for c, p in broken}
    for q in t.queries[: len(t.queries) // 2]:
        broken_pm[q] = t.nodes[1]["id"]
    broken = [(c, p) for c, p in broken_pm.items()]

    print("== 评测桥自检（合成树）==")
    for name, pe in [("完美预测", perfect), ("碎片化预测", broken)]:
        nf = node_f1(pe, gold)[2]
        ef = edge_f1(pe, gold)[2]
        af_macro = ancestor_f1(pe, gold, "macro")
        af_micro = ancestor_f1(pe, gold, "micro")[2]
        print(f"  {name:8s} Node F1={nf:.4f} Edge F1={ef:.4f} "
              f"Ancestor F1 macro={af_macro:.4f} micro={af_micro:.4f}")

    def ranked_for(pe):
        pm = _parent_map(pe)
        out = {}
        for q in t.queries:
            cand = t.negatives[q] + [t.gold_parent[q]]
            # 用「预测父」是否等于 gold 父来造排序，纯为验证指标同向
            cand = sorted(cand, key=lambda c: 0 if pm.get(q) == t.gold_parent[q]
                          and c == pm.get(q) else 1)
            out[q] = cand
        return out

    m_ok = taxobell_metrics(ranked_for(perfect), t.gold_parent)
    m_bad = taxobell_metrics(ranked_for(broken), t.gold_parent)
    print(f"  完美 MR={m_ok['MR']:.3f} MRR={m_ok['MRR']:.3f} | "
          f"碎片化 MR={m_bad['MR']:.3f} MRR={m_bad['MRR']:.3f}")

    a_ok, a_bad = ancestor_f1(perfect, gold), ancestor_f1(broken, gold)
    ok = (a_ok > a_bad) and (m_ok["MRR"] >= m_bad["MRR"]) and abs(a_ok - 1.0) < 1e-9
    print(f"  [{'PASS' if ok else 'FAIL'}] Ancestor F1 与 MRR 同向，且完美预测 Ancestor F1=1")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())
