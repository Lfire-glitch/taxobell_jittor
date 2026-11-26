"""BERT 端到端微调耗时预估：源码实证参数 + 降级档位 + CPU 预算判定"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import hard_negatives as HN  # noqa: E402

OUT = os.path.join(ROOT, "logs", "cost_estimate.json")
with open(os.path.join(ROOT, "config", "default.json"), encoding="utf-8") as f:
    CM = json.load(f)["cost_model"]
FACTS = dict(
    source="peel_ref/taxobell/src/{main,model,exp,data}.py 逐行核对",
    bert_params=110e6, proj_params=120e3, padmaxlen=30, batch_size=128,
    epochs_readme=50, epochs_appendix=125, accumulation_steps=1,
    negsamples=50, method_default="normal", drop_last=True,
    negs_by_dataset=dict(env=50, sci=50, wordnet=10, food=20, mesh=20),
    get_cls_calls_normal=2, get_cls_calls_gated=4, concepts_per_triple=3,
    bert_calls_per_step_normal=6, bert_calls_per_step_gated=12,
    optimizer="AdamW(all named_parameters) → BERT 全参微调",
    checkpoint_per_epoch=True, predict_per_epoch=True,
    dataset_edges_est=dict(env=261, sci=452, wordnet=19500, food=1533, mesh=10498),
    dataset_edges_note="docs/mainline.md §5；WordNet 20.5/19.5 按千计；待 .pkl 到位核实",
    triples_rule="edges × negsamples；.pkl 的 train_child_parent_negative_parent_triple 已离线展开",
    excluded="每 epoch predict() 推理 + checkpoint I/O 未计入，结果为下界",
)
EFF_DEFAULT = dict(CM["eff_gflops_assumed"])
EFF_NOTE = "CPU FP32 有效吞吐 GFLOPS 假设；用 --eff-gflops 或 WSL 实测标定覆盖"


def _fit(b, t):
    n = len(b)
    mb, mt = sum(b) / n, sum(t) / n
    s = sum((x - mb) * (y - mt) for x, y in zip(b, t))
    d = sum((x - mb) ** 2 for x in b)
    sl = s / d if d else 0.0
    return sl, mt - sl * mb


def step_time_model(cal_path):
    if not cal_path or not os.path.exists(cal_path):
        return None
    with open(cal_path, encoding="utf-8") as f:
        d = json.load(f)
    pts = sorted((int(k), v["median_s"]) for k, v in d["batches"].items())
    sl, ic = _fit([x for x, _ in pts], [y for _, y in pts])
    return dict(intercept_s=ic, per_sample_s=sl, points=dict(pts),
                max_measured_batch=max(x for x, _ in pts),
                note="t_step≈a+b·batch；实测 batch 上限外为线性外推；6/12 次调用按倍数折")


def measured_hours(r, stm):
    if r["frozen"]:
        return 0.0
    if not stm:
        return None
    t = stm["intercept_s"] + stm["per_sample_s"] * r["batch"]
    return r["total_steps"] * r["bert_calls"] * t / 3600


def bert_calls(t):
    if t["frozen"]:
        return 0
    return 3 * (4 if t["method"] == "gated" else 2)


def cost(t, eff, budget):
    triples = t.get("triples", t["edges"] * t["negs"])
    steps = math.ceil(triples / t["batch"])
    f_bert = bert_calls(t) * 6.0 * FACTS["bert_params"] * t["batch"] * t["seq"]
    f_proj = 6.0 * FACTS["proj_params"] * t["batch"] * 3
    f = f_bert + f_proj
    total = t["epochs"] * steps * f
    hours = {k: total / (v * 1e9) / 3600 for k, v in eff.items()}
    return dict(name=t["name"], frozen=t["frozen"], method=t["method"],
                edges=t["edges"], negs=t["negs"], batch=t["batch"],
                epochs=t["epochs"], seq=t["seq"], bert_calls=bert_calls(t),
                steps_per_epoch=steps, total_steps=steps * t["epochs"],
                flops_per_step=f, total_flops=total, hours=hours,
                within_budget=dict(at_least_one=min(hours.values()) <= budget,
                                   all=max(hours.values()) <= budget))


def tiers():
    E = FACTS["dataset_edges_est"]
    N = FACTS["negs_by_dataset"]
    full_triples = sum(E[k] * N[k] for k in E)
    b = dict(edges=E["env"], negs=50, batch=128, epochs=50, seq=30,
             method="normal", frozen=False)

    def T(name, **kw):
        d = dict(b)
        d.update(kw)
        d["name"] = name
        return d
    return [T("full_5ds_50ep", triples=full_triples, edges=sum(E.values())),
            T("full_5ds_125ep", triples=full_triples, edges=sum(E.values()), epochs=125),
            T("env_50ep"), T("env_50ep_batch32", batch=32),
            T("env_50ep_batch8", batch=8),
            T("env_5ep_negs10_batch32", epochs=5, negs=10, batch=32),
            T("env_2ep_negs5_batch8", epochs=2, negs=5, batch=8),
            T("probe_1ep_negs5_batch8", epochs=1, negs=5, batch=8),
            T("probe_1ep_gated", epochs=1, negs=5, batch=8, method="gated"),
            T("frozen_cached_cls_20ep", epochs=20, negs=5, batch=32, frozen=True)]


def sweep(eff, budget):
    out = []
    for b in (8, 16, 32, 64, 128):
        t = dict(name=f"batch{b}", edges=261, negs=10, batch=b, epochs=5,
                 seq=30, method="normal", frozen=False)
        out.append(cost(t, eff, budget))
    return out


def sampling_cost():
    p = os.path.join(ROOT, "logs", "negative_sampling.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    from src import synth
    return dict(source="in-process", cases=dict(
        synth=HN.timed(synth.build().edges), mesh_scale=HN.timed(HN.random_tree(10000))))


def resolve_eff(a):
    if a.calibrate_json:
        with open(a.calibrate_json, encoding="utf-8") as f:
            x = json.load(f)["eff_gflops"]
        return {f"calibrated({x:.1f}G)": x}, dict(mode="calibrated", path=a.calibrate_json)
    if a.eff_gflops:
        return {f"measured({a.eff_gflops:.1f}G)": a.eff_gflops}, dict(mode="measured")
    return EFF_DEFAULT, dict(mode="assumed", note=EFF_NOTE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=CM["budget_cpu_hours"])
    ap.add_argument("--eff-gflops", type=float, default=0.0)
    ap.add_argument("--calibrate-json", default="")
    a = ap.parse_args()
    eff, eff_meta = resolve_eff(a)
    stm = step_time_model(a.calibrate_json)
    rows = [cost(t, eff, a.budget) for t in tiers()]
    for r in rows:
        r["hours_measured"] = measured_hours(r, stm)
    full = rows[0]["total_flops"]
    for r in rows:
        r["ratio_to_full"] = r["total_flops"] / full if full else 0.0
    sw = sweep(eff, a.budget)
    sp = sampling_cost()
    spread = max(x["total_flops"] for x in sw) - min(x["total_flops"] for x in sw)
    spread_rel = spread / max(x["total_flops"] for x in sw)
    two, day = CM["budget_cpu_hours"], CM["budget_day_hours"]
    def hrs(r):
        return r["hours_measured"] if r["hours_measured"] is not None else max(r["hours"].values())
    verdict = dict(
        budget_hours=a.budget, budget_2h=two, budget_day_hours=day,
        full_hours=rows[0]["hours"], step_time_model=stm,
        basis="hours_measured（实测 step-time 外推）" if stm else "hours（FLOPs/eff 假设）",
        full_feasible_on_cpu=bool(hrs(rows[0]) <= day),
        feasible_tiers_2h=[r["name"] for r in rows if hrs(r) <= two],
        feasible_tiers_day=[r["name"] for r in rows if hrs(r) <= day],
        feasible_tiers_at_least_one=[r["name"] for r in rows if r["within_budget"]["at_least_one"]],
        feasible_tiers_all=[r["name"] for r in rows if r["within_budget"]["all"]],
        sampling_feasible=bool(sp["cases"]["mesh_scale"]["resample_all_s"] / 3600 <= two),
        sampling_hours=sp["cases"]["mesh_scale"]["resample_all_s"] / 3600,
        batch_only_helps_memory=bool(spread_rel < 0.05),
        batch_sweep_total_flops=[x["total_flops"] for x in sw],
        batch_sweep_spread_rel=spread_rel,
        batch_note=(f"FLOPs 总量近恒定（离散 {spread_rel:.2%}）；CPU 实测 "
                    f"a≈{stm['intercept_s']:.2f}s 固定 + b≈{stm['per_sample_s']:.3f}s/样本，"
                    f"实测 batch≤{stm['max_measured_batch']}，升 batch 摊薄才省时，降 batch 每样本更贵"
                    if stm else "FLOPs 总量近恒定；batch 只降峰值内存，不省算力"),
        frozen_ratio=next(r["ratio_to_full"] for r in rows if r["frozen"]),
        biggest_lever="冻结/缓存 [CLS]：BERT 调用 6→0，训练算力仅剩投影头",
        second_lever="epochs / 数据集边数 / negsamples / seq_len 均线性下降",
        sampling_is_offline=True,
        sampling_note="官方 .pkl 已展开三元组、__init__ 一次性 tokenize；无每 epoch 重挖",
        requires=["bert-base-uncased HF state_dict→npz（或本地 SciBERT/SPECTER 仅作架构标定）",
                  "≥1 个 processed/taxonomy_data_*.pkl", "tokenizer 词表"],
        excluded=FACTS["excluded"], eff_meta=eff_meta)
    doc = dict(facts=FACTS, hardware=eff, tiers=rows, batch_sweep=sw,
               sampling=sp, verdict=verdict)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"== BERT 微调耗时预估（2h={two} / day={day}h, eff={eff_meta['mode']}, "
          f"basis={verdict['basis']}）==")
    print(f"  {'tier':28s} {'steps':>8s} {'TFLOPs':>10s} {'ratio':>9s} "
          f"{'h_assum':>9s} {'h_meas':>9s}  2h/day")
    for r in rows:
        hm = r["hours_measured"]
        hs = f"{hm:9.2f}" if hm is not None else "        -"
        print(f"  {r['name']:28s} {r['total_steps']:8d} {r['total_flops']/1e12:10.2f} "
              f"{r['ratio_to_full']:9.2e} {max(r['hours'].values()):9.2f} {hs} "
              f"  {'Y' if hrs(r) <= two else '-'}/{'Y' if hrs(r) <= day else '-'}")
    print(f"  batch 8→128 总 FLOPs 近恒定={verdict['batch_only_helps_memory']} "
          f"(相对离散 {spread_rel:.2%}，仅 drop_last 余数；只降峰值内存)")
    print(f"  冻结/缓存档相对全量 ratio={verdict['frozen_ratio']:.3e}")
    if stm:
        print(f"  step-time 拟合 a={stm['intercept_s']:.2f}s(固定) + "
              f"b={stm['per_sample_s']:.3f}s/样本；实测 batch≤{stm['max_measured_batch']}，更大为外推")
    print(f"  2h 预算可跑={verdict['feasible_tiers_2h']}")
    print(f"  {day}h 预算可跑={verdict['feasible_tiers_day']}")
    print(f"  §4.4 重挖全 epoch={sp['cases']['mesh_scale']['resample_all_s']:.2f}s "
          f"({verdict['sampling_hours']:.2e}h) → 可做={verdict['sampling_feasible']}（官方为离线一次性）")
    print(f"  预算内档位(FLOPs 口径)={verdict['feasible_tiers_at_least_one']}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
