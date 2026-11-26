"""合成 mini-taxonomy 生成器（L0 数据层，纯 numpy / 标准库，无外部数据）"""
import hashlib
import json
import random
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class SynthTaxonomy:
    seed: int
    nodes: list              # [{"id": str, "name": str, "parent": str|None}]
    edges: list              # [[child, parent], ...]
    queries: list            # 留出叶节点 id
    negatives: dict          # query_id -> [candidate_parent_id, ...]（含真父 + hard negatives）
    gold_parent: dict        # query_id -> 真父 id


def build(seed=42, depth=4, branch=3, query_ratio=0.20, negs_per_query=5):
    rng = random.Random(seed)
    nodes, edges = [], []

    def grow(parent, d):
        nid = f"n{len(nodes):03d}"
        nodes.append({"id": nid, "name": f"concept_{nid}",
                      "parent": parent, "depth": d})
        if parent is not None:
            edges.append([nid, parent])
        if d < depth:
            for _ in range(rng.randint(2, branch)):
                grow(nid, d + 1)

    grow(None, 0)
    by_id = {n["id"]: n for n in nodes}
    children = {}
    for c, p in edges:
        children.setdefault(p, []).append(c)

    leaves = [n["id"] for n in nodes if n["id"] not in children and n["parent"] is not None]
    rng.shuffle(leaves)
    n_query = max(1, int(round(len(leaves) * query_ratio)))
    queries = sorted(leaves[:n_query])

    def ancestors(nid):
        out = []
        cur = by_id[nid]["parent"]
        while cur is not None:
            out.append(cur)
            cur = by_id[cur]["parent"]
        return out

    negatives, gold_parent = {}, {}
    for q in queries:
        p = by_id[q]["parent"]
        gold_parent[q] = p
        cand = []
        # siblings：同父的其他子节点
        cand += [c for c in children.get(p, []) if c != q]
        # uncles：父的兄弟
        gp = by_id[p]["parent"]
        if gp is not None:
            cand += [u for u in children.get(gp, []) if u != p]
            # cousins：uncle 的子节点
            for u in children.get(gp, []):
                cand += children.get(u, [])[:2]
        # grandparents：祖先中非真父者（最容易被误挂的硬负例）
        cand += [a for a in ancestors(q) if a != p][:2]
        cand = [c for c in dict.fromkeys(cand) if c != p]
        rng.shuffle(cand)
        negatives[q] = cand[:negs_per_query]

    return SynthTaxonomy(seed=seed, nodes=nodes, edges=edges,
                         queries=queries, negatives=negatives,
                         gold_parent=gold_parent)


def l0_fingerprint(t: SynthTaxonomy):
    """L0 契约：两端必须逐字一致的指纹。"""
    edges_json = json.dumps(sorted(t.edges), separators=(",", ":"))
    return {
        "seed": t.seed,
        "n_nodes": len(t.nodes),
        "n_edges": len(t.edges),
        "n_queries": len(t.queries),
        "query_ids_sha256": hashlib.sha256(
            json.dumps(sorted(t.queries)).encode()).hexdigest()[:16],
        "edges_sha256": hashlib.sha256(edges_json.encode()).hexdigest()[:16],
        "negs_sha256": hashlib.sha256(
            json.dumps(t.negatives, sort_keys=True).encode()).hexdigest()[:16],
    }


def export_triplets(t: SynthTaxonomy, dim=8, out_path=None):
    """导出固定张量给两个后端读同一份（消除框架 RNG 差异）。

    这里用「确定性伪嵌入」代替 BERT 输出：每个节点由 id 哈希出一个 d 维向量。
    降级记账里必须写明：真实跑法要换成缓存的 BERT [CLS] 嵌入（见 docs/mainline.md）。
    """
    g = np.random.default_rng(t.seed)
    ids = [n["id"] for n in t.nodes]
    emb = {}
    for nid in ids:
        h = hashlib.sha256(nid.encode()).digest()
        local = np.random.default_rng(int.from_bytes(h[:8], "little") % (2**32))
        emb[nid] = local.normal(0, 0.6, dim).astype(np.float32)

    rows = []
    for q in t.queries:
        for cand in t.negatives[q]:
            rows.append((q, t.gold_parent[q], cand))
    mu_c = np.stack([emb[c] for c, p, n in rows]).astype(np.float32)
    mu_p = np.stack([emb[p] for c, p, n in rows]).astype(np.float32)
    mu_n = np.stack([emb[n] for c, p, n in rows]).astype(np.float32)
    # o 由同一嵌入经确定性映射得到，父比子宽（结构性先验，便于损失方向检验）
    o_c = (0.6 + 0.15 * np.abs(mu_c)).astype(np.float32)
    o_p = (1.2 + 0.15 * np.abs(mu_p)).astype(np.float32)
    o_n = (0.9 + 0.15 * np.abs(mu_n)).astype(np.float32)

    # 降级态：冻结编码器的输出 n 直接取伪嵌入，投影头把 n 映射成 mu/o。
    # 算子层自检读 mu_*/o_*（已投影好的量），训练层读 n_*（投影头输入），互不干扰。
    data = dict(mu_c=mu_c, o_c=o_c, mu_p=mu_p, o_p=o_p,
                mu_n=mu_n, o_n=o_n,
                n_c=mu_c.copy(), n_p=mu_p.copy(), n_n=mu_n.copy(),
                row_ids=np.array([f"{c}|{p}|{n}" for c, p, n in rows], dtype=object))
    if out_path:
        np.savez(out_path, **data)
    return data


def to_json(t: SynthTaxonomy):
    return json.dumps(asdict(t), ensure_ascii=False, indent=2)
