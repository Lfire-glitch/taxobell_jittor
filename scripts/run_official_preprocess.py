"""驱动官方 pre_process 生成 processed/*.pkl（不改官方逻辑）"""
import argparse
import importlib.util
import os
import shutil
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(ROOT, "chk", "official")
DRIVE = os.path.join(ROOT, "data", "official_drive", "data")
QREPO = os.path.join(ROOT, "data", "official_raw")
MULTI = {"mesh", "semeval_food", "computer_science", "psychology",
         "wordnet_verb"}
SINGLE_REQ = ("{ds}_raw_en.taxo", "{ds}_train.taxo", "dic.json",
              "{ds}_eval.terms", "{ds}_eval.gt")
MULTI_REQ = ("{ds}.terms", "{ds}.taxo", "{ds}_train.taxo",
             "{ds}_test.terms", "defs.json", "key_value.json",
             "test_taxo.json")


def load_official():
    try:
        import tqdm  # noqa: F401
    except ImportError:
        sys.modules["tqdm"] = type(sys)("tqdm")
        sys.modules["tqdm"].tqdm = lambda it, **k: it
    spec = importlib.util.spec_from_file_location(
        "tb_official_pre", os.path.join(OFFICIAL, "pre_process.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def source_dir(ds):
    for base in (DRIVE, QREPO):
        p = os.path.join(base, ds)
        if os.path.isdir(p):
            return p
    raise SystemExit("找不到 %s 的原始数据（检查 data/official_drive/data）" % ds)


def stage(ds, kind):
    src = source_dir(ds)
    req = SINGLE_REQ if kind == "single" else MULTI_REQ
    for pat in req:
        f = pat.format(ds=ds)
        if not os.path.exists(os.path.join(src, f)):
            raise SystemExit("缺 %s/%s" % (ds, f))
    dst = os.path.join(ROOT, "data", ds)
    os.makedirs(os.path.join(dst, "processed"), exist_ok=True)
    for pat in req:
        f = pat.format(ds=ds)
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["environment", "science", "semeval_food", "mesh"])
    ap.add_argument("--expid", type=int, default=0)
    ap.add_argument("--negsamples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    import numpy as np
    mod = load_official()
    os.makedirs(os.path.join(ROOT, "preprocess_work"), exist_ok=True)
    os.chdir(os.path.join(ROOT, "preprocess_work"))
    for ds in a.datasets:
        kind = "multi" if ds in MULTI else "single"
        src = stage(ds, kind)
        out = os.path.join(ROOT, "data", ds, "processed",
                           "taxonomy_data_%d%d_.pkl" % (a.expid, a.negsamples))
        if os.path.exists(out) and not a.force:
            print("[skip] %s（--force 覆盖）" % os.path.relpath(out, ROOT))
            continue
        np.random.seed(a.seed)
        args = SimpleNamespace(dataset=ds, expID=a.expid,
                               negsamples=a.negsamples)
        print("[run ] %-14s kind=%-6s src=%s" % (ds, kind,
              os.path.relpath(src, ROOT)))
        (mod.create_multiparent_data if kind == "multi"
         else mod.create_data)(args)
        print("[done] %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
