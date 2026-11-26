"""B 档端到端骨架：bert-base-uncased [CLS] → 双 MLP → 双损失 → 训练 → 评测桥"""
import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("cc_path", "/usr/bin/g++-12")

import numpy as np
from src import ops_numpy as O
from src import projection, synth
from src.bert_tokenizer import BertTokenizer

EXT = os.path.join(ROOT, "data", "external", "bert-base-uncased")
DEFAULT_REPO = "/home/guo76/specter-jittor"


def load_old(repo):
    def mod(name, rel):
        p = os.path.join(repo, rel)
        if not os.path.exists(p):
            raise SystemExit("缺旧仓 %s；用 --repo 指定 specter-jittor 路径" % p)
        spec = importlib.util.spec_from_file_location(name, p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    hf = mod("tb_old_hf", "src/model/hf_loader.py")
    return mod("tb_old_bert", "src/model/bert.py").BertModel, hf.load_npz, hf.build_mapping


def dry_run_data():
    t = synth.build()
    names = [n["name"] for n in t.nodes]
    defs = ["concept %s at depth %d in the seed taxonomy"
            % (n["id"], n["depth"]) for n in t.nodes]
    idx = {n["id"]: i for i, n in enumerate(t.nodes)}
    rows = [(q, t.gold_parent[q], c) for q in t.queries for c in t.negatives[q]]
    return t, names, defs, idx, rows


def _text(x):
    if isinstance(x, str):
        return x, ""
    if isinstance(x, (tuple, list)):
        return str(x[0]), str(x[1]) if len(x) > 1 else ""
    if isinstance(x, dict):
        return (str(x.get("name", x.get("term", ""))),
                str(x.get("def", x.get("definition", ""))))
    return str(x), ""


def load_pkl_triplets(path):
    """官方 12 键结构（chk/official/datasets.py 实证）：ID 三元组 + id2concept/id2context。"""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "train_child_parent_negative_parent_triple" in obj:
        return load_official_pkl(obj)
    keys = list(obj) if isinstance(obj, dict) else []
    cand = None
    for k in ("train_child_parent_negative_parent_triple",
              "train_child_parent_negative_parent_triples",
              "train_triplets", "train", "triplets"):
        if isinstance(obj, dict) and k in obj:
            cand = obj[k]
            break
    if cand is None and isinstance(obj, (list, tuple)):
        cand = obj
    if cand is None:
        raise SystemExit(
            "无法识别 pkl 结构，keys=%r\n"
            "把该 keys 贴给智能体补解析分支。" % (keys[:20],))
    names, defs, idx, rows, gold, negs = [], [], [], [], {}, {}

    def intern(x):
        key, _ = _text(x)
        if key not in idx:
            s, d = _text(x)
            idx[key] = len(names)
            names.append(s)
            defs.append(d or ("taxonomy concept " + key))
        return key

    for tri in cand:
        if len(tri) < 3:
            continue
        c, p, n = (intern(tri[0]), intern(tri[1]), intern(tri[2]))
        rows.append((c, p, n))
        gold[c] = p
        negs.setdefault(c, [])
        if n not in negs[c]:
            negs[c].append(n)
    if not rows:
        raise SystemExit("pkl 里没读到任何三元组，检查 %s 的内容" % path)
    t = SimpleNamespace(edges=[[c, p] for c, p in gold.items()],
                        queries=list(gold), gold_parent=gold, negatives=negs,
                        partial_taxonomy=True)
    return t, names, defs, idx, rows


def load_official_pkl(obj):
    """ID→concept/context 解析：官方消费代码只用 id2context 喂 tokenizer，
    id2concept 是名字，边取 child_parent_pair（单父 path2root[0] 是自身，勿用），
    test_* 是留出 query；三元组含 np.int64（官方 np.random.choice 产物）须归一化。"""
    id2name, id2ctx = obj.get("id2concept", {}), obj.get("id2context", {})

    def resolve(x):
        # 官方 np.random.choice 会把 np.int64 直接塞进三元组；它不是 int 子类，
        # 不归一化会被 str() 注册成 "7277" 这类伪概念，污染候选池与 BERT 文本。
        if isinstance(x, np.integer):
            x = int(x)
        if isinstance(x, (int, str)) and x in id2name:
            return str(id2name[x]), str(id2ctx.get(x, ""))
        return _text(x)

    names, defs, idx = [], [], {}

    def intern(x):
        key, ctx = resolve(x)
        if key not in idx:
            idx[key] = len(names)
            names.append(key)
            defs.append(ctx or ("taxonomy concept " + key))
        return key

    for cid in sorted(obj.get("concept_set", ())):
        intern(cid)
    trip = obj["train_child_parent_negative_parent_triple"]
    rows, gold, negs = [], {}, {}
    for tri in trip:
        if len(tri) < 3:
            continue
        c, p, n = (intern(tri[0]), intern(tri[1]), intern(tri[2]))
        rows.append((c, p, n))
        gold[c] = p
        negs.setdefault(c, [])
        if n not in negs[c]:
            negs[c].append(n)

    queries, gold_test = [], {}
    for q, g in zip(obj.get("test_concepts_id", ()), obj.get("test_gt_id", ())):
        if isinstance(g, (list, tuple, set, frozenset)):
            g = sorted(g)[0]   # 多父数据集取首个 gold（口径记入 note）
        qn, gn = intern(q), intern(g)
        queries.append(qn)
        gold_test[qn] = gn
    if not queries:
        queries, gold_test = list(gold), gold

    p2r = obj.get("path2root", {})
    # child_parent_pair 是 [child, parent] 权威边；单父 path2root[v]=[v,parent,...,root]
    # 首元素是自身，取 [1] 才是真父；multiparent 版 path2root 为空，全靠 child_parent_pair。
    if p2r:
        edges = [[intern(cid), intern(pp[1])] for cid, pp in p2r.items()
                 if len(pp) > 1]
    else:
        edges = [[intern(c), intern(p)]
                 for c, p in obj.get("child_parent_pair", [])]
    edges += [[c, p] for c, p in gold_test.items()]
    cand_keys = [n for n in names if n not in set(queries)]
    t = SimpleNamespace(edges=edges, queries=queries, gold_parent=gold_test,
                        negatives=negs, partial_taxonomy=True,
                        cand_keys=cand_keys)
    return t, names, defs, idx, rows


def encode_frozen(jt, bert, tok, names, defs, seq_len, batch):
    """冻结前向：eval() 关 dropout，一次编码并缓存 [CLS]（推理类才 eval()）。"""
    bert.eval()
    out = []
    with jt.no_grad():
        for s in range(0, len(names), batch):
            ids, mask, tt = tok.encode_batch(names[s:s + batch],
                                             defs[s:s + batch], seq_len)
            h, _ = bert(jt.array(np.asarray(ids, np.int32)),
                        jt.array(np.asarray(mask, np.float32)),
                        jt.array(np.asarray(tt, np.int32)))
            jt.sync_all()
            out.append(np.asarray(h[:, 0].numpy(), np.float32))
    return np.concatenate(out, 0)


def build_batches(cls_all, rows, idx, batch, seed=42):
    order = np.random.default_rng(seed).permutation(len(rows))
    out = []
    for s in range(0, len(order), batch):
        trip = [rows[i] for i in order[s:s + batch]]
        out.append({
            "n_c": cls_all[[idx[c] for c, p, n in trip]],
            "n_p": cls_all[[idx[p] for c, p, n in trip]],
            "n_n": cls_all[[idx[n] for c, p, n in trip]],
        })
    return out


def init_head(cls_dim, hidden, dim):
    from src import model_jittor as M
    w = projection.init_weights(k=cls_dim, hidden=hidden, d=dim)
    head = M.TaxoBellProjection(w)
    head.train()
    return head, w


def step_common(jt, head, opt, batch, loss_name):
    jb = {k: jt.array(v) if isinstance(v, np.ndarray) else v
          for k, v in batch.items()}
    out = head.loss(jb, loss_name)
    opt.backward(out["total"])
    rec = {k: float(v.item()) for k, v in out.items()}
    opt.step()
    jt.sync_all()
    return rec


def run_frozen(jt, cls_all, rows, idx, a):
    head, w0 = init_head(cls_all.shape[1], a.hidden, a.dim)
    opt = jt.optim.AdamW(head.param_list(), lr=a.lr)
    batches = build_batches(cls_all, rows, idx, a.batch)
    t0, curve = time.perf_counter(), []
    for s in range(a.steps):
        curve.append(step_common(jt, head, opt, batches[s % len(batches)], a.loss))
    w = {k: head.params[k].numpy() for k in projection.WEIGHT_KEYS}
    return w, w0, curve, time.perf_counter() - t0, len(projection.WEIGHT_KEYS)


def run_finetune(jt, bert, build_mapping, tok, names, defs, rows, idx, a):
    head, w0 = init_head(768, a.hidden, a.dim)
    params = list(build_mapping(bert).values()) + head.param_list()
    opt = jt.optim.AdamW(params, lr=a.lr)
    g = np.random.default_rng(a.seed)
    t0, curve = time.perf_counter(), []
    for s in range(a.steps):
        sel = g.choice(len(rows), min(a.batch, len(rows)), replace=False)
        trip = [rows[i] for i in sel]
        ns, ds = [], []
        for c, p, n in trip:
            for key in (c, p, n):
                ns.append(names[idx[key]])
                ds.append(defs[idx[key]])
        ids, mask, tt = tok.encode_batch(ns, ds, a.seq_len)
        h, _ = bert(jt.array(np.asarray(ids, np.int32)),
                    jt.array(np.asarray(mask, np.float32)),
                    jt.array(np.asarray(tt, np.int32)))
        B = len(trip)
        cls = h[:, 0]
        batch = {"n_c": cls[0:B], "n_p": cls[B:2 * B], "n_n": cls[2 * B:3 * B]}
        curve.append(step_common(jt, head, opt, batch, a.loss))
    w = {k: head.params[k].numpy() for k in projection.WEIGHT_KEYS}
    return w, w0, curve, time.perf_counter() - t0, len(params)


def load_eval_bridge():
    p = os.path.join(ROOT, "scripts", "eval_bridge.py")
    spec = importlib.util.spec_from_file_location("tb_eval_bridge", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def evaluate(t, cls_all, rows, idx, w):
    ev = load_eval_bridge()
    mu, o = projection.forward_numpy(w, cls_all)
    ranked = {"BC": {}, "KL": {}}
    pool = list(getattr(t, "cand_keys", []))
    for q in t.queries:
        g = t.gold_parent.get(q)
        if pool:
            cand = [c for c in pool if c != q]
            if g not in cand:
                cand = cand + [g]
        else:
            ii = [i for i, (c, p, n) in enumerate(rows) if c == q]
            if not ii:
                continue
            cand = [g] + [rows[i][2] for i in ii]
        m = np.stack([mu[idx[x]] for x in cand]).astype(np.float32)
        v = np.stack([o[idx[x]] for x in cand]).astype(np.float32)
        qm, qo = mu[idx[q]][None], o[idx[q]][None]
        for mode in ("BC", "KL"):
            s = O.rank_anchors(qm, qo, m, v, mode)[0]
            ranked[mode][q] = [cand[i] for i in np.argsort(-s)]

    def scen(r):
        gold = [tuple(e) for e in t.edges]
        pm = {c: p for c, p in t.edges}
        pm.update({q: r[q][0] for q in r})
        pe = [(c, p) for c, p in pm.items()]
        met = ev.taxobell_metrics(r, t.gold_parent)
        return dict(Node_F1=ev.node_f1(pe, gold)[2],
                    Edge_F1=ev.edge_f1(pe, gold)[2],
                    Ancestor_F1_macro=ev.ancestor_f1(pe, gold, "macro"),
                    **met)

    return {mode: scen(ranked[mode]) for mode in ("BC", "KL")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="")
    ap.add_argument("--npz", default=os.path.join(EXT, "bert_jittor.npz"))
    ap.add_argument("--vocab", default=os.path.join(EXT, "vocab.txt"))
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--loss", choices=("paper", "official"), default="paper")
    ap.add_argument("--finetune-bert", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--save-weights", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    mode = "pkl" if a.pkl else "dry_run"
    for p in (a.npz, a.vocab, os.path.join(a.repo, "src/model/bert.py")):
        if not os.path.exists(p):
            print("缺 %s；先在 Windows 侧跑 scripts/convert_bert_safetensors.py 并同步" % p)
            return 1
    import jittor as jt
    jt.flags.use_cuda = 0
    BertModel, load_npz, build_mapping = load_old(a.repo)
    tok = BertTokenizer(a.vocab, max_len=a.seq_len)
    bert = BertModel(vocab_size=tok.vocab_size)
    load_npz(bert, a.npz, verbose=False)
    t, names, defs, idx, rows = (load_pkl_triplets(a.pkl) if a.pkl else dry_run_data())
    print("== B 档端到端 ==")
    print("  mode=%s finetune_bert=%s loss=%s steps=%d batch=%d seq=%d"
          % (mode, a.finetune_bert, a.loss, a.steps, a.batch, a.seq_len))
    print("  nodes=%d triples=%d vocab=%d" % (len(names), len(rows), tok.vocab_size))

    t0 = time.perf_counter()
    if a.finetune_bert:
        w, w0, curve, train_s, nparam = run_finetune(
            jt, bert, build_mapping, tok, names, defs, rows, idx, a)
        cls_all = encode_frozen(jt, bert, tok, names, defs, a.seq_len, a.batch)
    else:
        cls_all = encode_frozen(jt, bert, tok, names, defs, a.seq_len, a.batch)
        w, w0, curve, train_s, nparam = run_frozen(jt, cls_all, rows, idx, a)
    encode_s = time.perf_counter() - t0 - train_s
    finite = all(np.isfinite(c["total"]) for c in curve)
    print("  encode=%.2fs train=%.2fs (%.3fs/step)  loss %.6f -> %.6f  finite=%s"
          % (encode_s, train_s, train_s / max(1, a.steps),
             curve[0]["total"], curve[-1]["total"], finite))

    wp = a.save_weights or os.path.join(
        ROOT, "logs", "b_e2e_%s%s.npz" % (mode, "_" + a.tag if a.tag else ""))
    np.savez(wp, **{k: np.asarray(v, np.float32) for k, v in w.items()})
    print("  -> %s" % os.path.relpath(wp, ROOT).replace("\\", "/"))

    evaluation = None
    if not a.skip_eval:
        try:
            evaluation = evaluate(t, cls_all, rows, idx, w)
            for m, r in evaluation.items():
                print("  [%s] MR=%.3f MRR=%.3f EdgeF1=%.4f AncF1=%.4f"
                      % (m, r["MR"], r["MRR"], r["Edge_F1"], r["Ancestor_F1_macro"]))
        except Exception as e:
            print("  [WARN] 评测桥跳过：%s: %s" % (type(e).__name__, e))
            evaluation = dict(error="%s: %s" % (type(e).__name__, e))

    name = "b_e2e_%s%s.json" % (mode, "_" + a.tag if a.tag else "")
    rec = dict(mode=mode, finetune_bert=a.finetune_bert, loss=a.loss,
               steps=a.steps, lr=a.lr, batch=a.batch, seq_len=a.seq_len,
               proj_in_dim=int(cls_all.shape[1]), proj_dim=a.dim,
               hidden=a.hidden, vocab_size=tok.vocab_size, seed=a.seed,
               n_nodes=len(names), n_triples=len(rows),
               n_optimizer_params=int(nparam),
               weight_fingerprint_init=projection.weight_fingerprint(w0),
               weight_fingerprint_trained=projection.weight_fingerprint(w),
               total=[c["total"] for c in curve], first=curve[0], last=curve[-1],
               finite=finite, encode_s=encode_s, train_s=train_s,
               per_step_s=train_s / max(1, a.steps),
               save_weights=os.path.relpath(wp, ROOT).replace("\\", "/"),
               evaluation=evaluation,
               note=("dry_run 用合成树；pkl 分支按官方 datasets.py 口径解析"
                     "（ID 三元组 + id2context，候选池 = concept_set − test，"
                     "多父 gold 取首个，边取 child_parent_pair）。冻结模式只训投影头，"
                     "微调模式含 BERT 全参；均非论文 5ds×50ep，不比论文绝对分数；"
                     "本地替代 pkl 是真实概念名+人工定义，非官方数据；"
                     "multiparent pkl 的 path2root 为空，Ancestor F1 用训练边近似。"))
    path = os.path.join(ROOT, "logs", name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print("  -> logs/%s" % name)
    print("  评测指标内置于本脚本（evaluation 字段）；本权重是 768 维输入，"
          "不能喂给 8 维降级口径的 eval_pipeline.py")
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main())
