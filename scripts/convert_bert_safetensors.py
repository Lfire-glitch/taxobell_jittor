"""hf-mirror 的 BertForMaskedLM safetensors → 旧仓 hf_loader.load_npz() 可装载的 npz"""
import argparse
import hashlib
import json
import os
import struct
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.join(ROOT, "data", "external", "bert-base-uncased", "model.safetensors")
DEFAULT_OUT = os.path.join(ROOT, "data", "external", "bert-base-uncased", "bert_jittor.npz")


def expected(vocab=30522, layers=12, h=768, inter=3072, pos=512):
    """复刻 hf_loader.build_mapping() 的 199 键与 shape，不 import jittor。"""
    e = {}
    e["embeddings.word_embeddings.weight"] = (vocab, h)
    e["embeddings.position_embeddings.weight"] = (pos, h)
    e["embeddings.token_type_embeddings.weight"] = (2, h)
    e["embeddings.LayerNorm.weight"] = (h,)
    e["embeddings.LayerNorm.bias"] = (h,)
    for i in range(layers):
        p = "encoder.layer.%d." % i
        for n in ("query", "key", "value"):
            e[p + "attention.self." + n + ".weight"] = (h, h)
            e[p + "attention.self." + n + ".bias"] = (h,)
        e[p + "attention.output.dense.weight"] = (h, h)
        e[p + "attention.output.dense.bias"] = (h,)
        e[p + "attention.output.LayerNorm.weight"] = (h,)
        e[p + "attention.output.LayerNorm.bias"] = (h,)
        e[p + "intermediate.dense.weight"] = (inter, h)
        e[p + "intermediate.dense.bias"] = (inter,)
        e[p + "output.dense.weight"] = (h, inter)
        e[p + "output.dense.bias"] = (h,)
        e[p + "output.LayerNorm.weight"] = (h,)
        e[p + "output.LayerNorm.bias"] = (h,)
    e["pooler.dense.weight"] = (h, h)
    e["pooler.dense.bias"] = (h,)
    return e


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, m in hdr.items():
            if k == "__metadata__":
                continue
            if m["dtype"] != "F32":
                raise SystemExit("仅支持 F32，遇到 %s: %s" % (k, m["dtype"]))
            s, e = m["data_offsets"]
            f.seek(base + s)
            out[k] = (tuple(m["shape"]), f.read(e - s))
        return out, hdr.get("__metadata__"), n


def remap(k):
    if k.startswith("cls."):
        return None
    if k.startswith("bert."):
        k = k[5:]
    if k.endswith(".gamma"):
        k = k[:-5] + "weight"
    elif k.endswith(".beta"):
        k = k[:-4] + "bias"
    return k


def npy_bytes(shape, raw):
    d = ("{'descr': '<f4', 'fortran_order': False, 'shape': %s, }"
         % repr(tuple(shape))).encode("ascii")
    pad = (-(10 + len(d) + 1)) % 16
    h = d + b" " * pad + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(h)) + h + raw


def verify_npz(path):
    """zipfile 读回 npy header，确认 np.load 能认、shape 与 mapping 一致。"""
    with zipfile.ZipFile(path) as z:
        names = [n[:-4] for n in z.namelist() if n.endswith(".npy")]
        assert len(names) == len(set(names)), "npz 内键名重复"
        got = {}
        for n in z.namelist():
            if not n.endswith(".npy"):
                continue
            b = z.read(n)
            assert b[:6] == b"\x93NUMPY", "npy magic 错 " + n
            hl = struct.unpack("<H", b[8:10])[0]
            hdr = b[10:10 + hl].decode("ascii")
            assert "'<f4'" in hdr and "False" in hdr, "npy header 错 " + n
            shape = tuple(int(x) for x in hdr.split("'shape': (")[1].split(")")[0].split(",") if x.strip())
            got[n[:-4]] = (shape, len(b) - 10 - hl)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--vocab", type=int, default=30522)
    ap.add_argument("--layers", type=int, default=12)
    a = ap.parse_args()
    if not os.path.exists(a.src):
        print("缺 %s，先跑 B 前置下载（hf-mirror.com）" % a.src)
        return 1
    tensors, meta, hdr_n = read_safetensors(a.src)
    exp = expected(vocab=a.vocab, layers=a.layers)
    mapped, dropped = {}, []
    for k, (shape, raw) in tensors.items():
        nk = remap(k)
        if nk is None:
            dropped.append(k)
            continue
        if nk in mapped:
            raise SystemExit("映射后键名冲突: " + nk)
        mapped[nk] = (shape, raw)
    missing = sorted(set(exp) - set(mapped))
    extra = sorted(set(mapped) - set(exp))
    bad_shape = [(k, mapped[k][0], exp[k]) for k in exp
                 if k in mapped and tuple(mapped[k][0]) != tuple(exp[k])]
    print("== safetensors → jittor npz ==")
    print("  header=%dB  tensors=%d  metadata=%s" % (hdr_n, len(tensors), meta))
    print("  dropped cls.* = %d  mapped=%d  expected=%d" % (len(dropped), len(mapped), len(exp)))
    print("  missing=%d extra=%d shape_mismatch=%d" % (len(missing), len(extra), len(bad_shape)))
    if missing or extra or bad_shape:
        print("  missing:", missing[:6])
        print("  extra  :", extra[:6])
        print("  bad    :", bad_shape[:6])
        return 1
    assert len(mapped) == 199, "键数应为 199，实际 %d" % len(mapped)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as z:
        for k in sorted(mapped):
            shape, raw = mapped[k]
            n = 1
            for d in shape:
                n *= d
            assert len(raw) == 4 * n, "字节数不符 %s %s" % (k, shape)
            z.writestr(k + ".npy", npy_bytes(shape, raw))
    got = verify_npz(a.out)
    assert set(got) == set(exp), "npz 读回键名与 mapping 不一致"
    for k, (shape, nb) in got.items():
        n = 1
        for d in exp[k]:
            n *= d
        assert shape == exp[k] and nb == 4 * n, "npz 读回 shape 不符 " + k
    with open(a.src, "rb") as f:
        src_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    rec = dict(source=os.path.relpath(a.src, ROOT).replace("\\", "/"),
               source_sha256_16=src_sha,
               output=os.path.relpath(a.out, ROOT).replace("\\", "/"),
               output_bytes=os.path.getsize(a.out),
               tensors_in=len(tensors), tensors_out=len(mapped),
               dropped_cls=len(dropped), missing=0, extra=0,
               shape_mismatch=0, transposed=0,
               note=("去 bert. 前缀 / gamma,beta→weight,bias / 丢 cls.*；"
                     "shape 与 hf_loader.build_mapping() 逐项一致，无需转置；"
                     "pooler.dense.* 已存在，无需补；BertModel 须 vocab_size=%d" % a.vocab))
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "logs", "bert_convert.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print("  -> %s (%d bytes)" % (rec["output"], rec["output_bytes"]))
    print("  -> logs/bert_convert.json")
    print("  下一步（WSL）：python scripts/check_bert_load.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
