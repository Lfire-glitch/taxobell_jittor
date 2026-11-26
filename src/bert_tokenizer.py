"""bert-base-uncased WordPiece，纯标准库；不依赖 transformers"""
import unicodedata


def _ctl(c):
    return unicodedata.category(c).startswith("C")


def _punct(c):
    o = ord(c)
    return (unicodedata.category(c).startswith("P")
            or 33 <= o <= 47 or 58 <= o <= 64 or 91 <= o <= 96 or 123 <= o <= 126)


def _cjk(c):
    o = ord(c)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF)


class BertTokenizer:
    def __init__(self, vocab_path, do_lower_case=True, max_len=128):
        with open(vocab_path, encoding="utf-8") as f:
            self.inv = [l.rstrip("\r\n") for l in f]
        self.vocab = {t: i for i, t in enumerate(self.inv)}
        self.lower = do_lower_case
        self.max_len = max_len
        self.pad, self.unk, self.cls, self.sep = (
            self.vocab[t] for t in ("[PAD]", "[UNK]", "[CLS]", "[SEP]"))
        self.vocab_size = len(self.inv)

    def _basic(self, text):
        buf = []
        for c in unicodedata.normalize("NFD", text):
            if ord(c) == 0xFFFD or _ctl(c):
                continue
            if _punct(c) or _cjk(c):
                buf.append(" " + c + " ")
            elif c.isspace():
                buf.append(" ")
            else:
                buf.append(c)
        s = "".join(buf)
        if self.lower:
            s = "".join(c for c in unicodedata.normalize("NFD", s.lower())
                       if unicodedata.category(c) != "Mn")
        return s.strip().split()

    def _wp(self, word):
        if word in self.vocab:
            return [word]
        cs, out, st = list(word), [], 0
        while st < len(cs):
            en, cur = len(cs), None
            while st < en:
                sub = "".join(cs[st:en])
                if st:
                    sub = "##" + sub
                if sub in self.vocab:
                    cur = sub
                    break
                en -= 1
            if cur is None:
                return ["[UNK]"]
            out.append(cur)
            st = en
        return out

    def _ids(self, text):
        return [self.vocab.get(t, self.unk)
                for w in self._basic(text) for t in self._wp(w)]

    def encode_pair(self, name, definition):
        a = [self.cls] + self._ids(name) + [self.sep]
        b = self._ids(definition) + [self.sep]
        room = self.max_len - len(a)
        if room <= 0:
            a, b = a[:self.max_len - 1] + [self.sep], []
        elif len(b) > room:
            b = b[:room - 1] + [self.sep]
        return a + b, [0] * len(a) + [1] * len(b)

    def encode_batch(self, names, defs, max_len=None):
        seqs = [self.encode_pair(n, d) for n, d in zip(names, defs)]
        L = min(max(len(s[0]) for s in seqs), max_len or self.max_len)
        ids, mask, tt = [], [], []
        for s, t in seqs:
            n = min(len(s), L)
            ids.append(s[:n] + [self.pad] * (L - n))
            mask.append([1.0] * n + [0.0] * (L - n))
            tt.append(t[:n] + [0] * (L - n))
        return ids, mask, tt


if __name__ == "__main__":
    import os
    import sys
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "external", "bert-base-uncased", "vocab.txt")
    if not os.path.exists(p):
        print("缺 vocab.txt，先跑 B 前置下载")
        sys.exit(1)
    tok = BertTokenizer(p)
    ids, mask, tt = tok.encode_batch(
        ["machine learning", "taxonomy induction", "卷积神经网络"],
        ["a field of study on statistical models",
         "constructing a hierarchy of concepts from text",
         "一类含卷积运算的神经网络"])
    print("vocab_size", tok.vocab_size, "shape", len(ids), len(ids[0]))
    for i, (a, m, t) in enumerate(zip(ids, mask, tt)):
        print(i, "len", sum(m), "head", a[:4], "tail", a[int(sum(m)) - 2:int(sum(m))],
              "seg", t[:6])
    assert ids[0][0] == tok.cls and tok.sep in ids[0]
    assert all(len(x) == len(ids[0]) for x in ids)
    print("tokenizer OK")
