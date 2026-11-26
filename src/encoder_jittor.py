"""编码器层：冻结 BERT 的 Jittor 前向 ↔ HF/PyTorch 逐层对照（TaxoBell §4.2）"""
import os
import sys

import numpy as np

DEFAULT_REPO = "/home/guo76/specter-jittor"
DEFAULT_VOCAB = 31116


def _import_jittor_bert(repo, vocab_size, weights_npz):
    os.environ.setdefault("cc_path", "/usr/bin/g++-12")  # 必须在 import jittor 之前
    import jittor as jt
    jt.flags.use_cuda = 0
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.model.bert import BertModel
    from src.model.hf_loader import load_npz
    model = BertModel(vocab_size=vocab_size)
    load_npz(model, weights_npz, verbose=False)
    model.eval()                 # 推理类才 eval()；下游训练态绝不 eval()
    return jt, model


def _tokenize(tokenizer_dir, names, defs, max_len):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    enc = tok(list(names), list(defs), padding=True, truncation="only_second",
              max_length=max_len, return_tensors="np")
    return (enc["input_ids"].astype("int32"),
            enc["attention_mask"].astype("float32"),
            enc["token_type_ids"].astype("int32"))


def encode_jittor(names, defs, weights_npz, tokenizer_dir,
                  repo=None, vocab_size=DEFAULT_VOCAB, max_len=128):
    """冻结 Jittor BERT → `[CLS]`，返回 numpy float32 [B,768]。"""
    jt, model = _import_jittor_bert(repo or DEFAULT_REPO, vocab_size, weights_npz)
    ids, mask, tt = _tokenize(tokenizer_dir, names, defs, max_len)
    with jt.no_grad():
        h, _ = model(jt.array(ids), jt.array(mask), jt.array(tt))
        jt.sync_all()
        return np.ascontiguousarray(h[:, 0].numpy(), dtype=np.float32)


def encode_hf(names, defs, model_dir, max_len=128):
    """HF / PyTorch 对照：`[CLS]` 最后层隐藏态，返回 numpy float32 [B,768]。"""
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    enc = tok(list(names), list(defs), padding=True, truncation="only_second",
              max_length=max_len, return_tensors="pt")
    ref = AutoModel.from_pretrained(model_dir, local_files_only=True).eval()
    with torch.no_grad():
        out = ref(**enc, output_hidden_states=True)
    return out.hidden_states[-1][:, 0].numpy().astype(np.float32)


def cls_parity(jt_cls, hf_cls):
    """Jittor ↔ HF 的 `[CLS]` 逐行余弦（取均值）+ max|Δ|；判据沿用旧仓库 03_m1_align.py。"""
    a = jt_cls.astype(np.float64)
    b = hf_cls.astype(np.float64)
    per_row = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1)
                                  * np.linalg.norm(b, axis=1) + 1e-12)
    return dict(cos_mean=float(per_row.mean()),
                cos_min=float(per_row.min()),
                max_abs_delta=float(np.abs(a - b).max()),
                shape=list(jt_cls.shape))


def cached_embeddings(npz_path):
    """降级路径：读 logs/parity_input.npz 里的 n_*，不触发任何框架。"""
    with np.load(npz_path, allow_pickle=True) as z:
        return {k: z[k].astype(np.float32) for k in ("n_c", "n_p", "n_n")}
