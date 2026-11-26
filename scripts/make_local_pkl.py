"""本地替代 pkl 生成器：严格按官方 12 键结构（chk/official/datasets.py 实证）"""
import argparse
import os
import pickle
import random

# (name, parent, gloss)
TAXO = [
    ("entity", None, "anything that exists"),
    ("object", "entity", "a physical entity that can be observed"),
    ("artifact", "object", "a man-made object"),
    ("instrument", "artifact", "a device that requires skill for proper use"),
    ("tool", "instrument", "an implement used in the practice of a vocation"),
    ("weapon", "instrument", "any instrument used in fighting or hunting"),
    ("utensil", "instrument", "an implement for practical use in cooking or eating"),
    ("conveyance", "artifact", "something that serves as a means of transportation"),
    ("vehicle", "conveyance", "a conveyance that transports people or objects"),
    ("aircraft", "conveyance", "a vehicle capable of atmospheric flight"),
    ("watercraft", "conveyance", "a vehicle designed for travel on water"),
    ("structure", "artifact", "a thing constructed of parts joined together"),
    ("building", "structure", "a structure with walls and a roof"),
    ("bridge", "structure", "a structure that allows crossing over an obstacle"),
    ("tower", "structure", "a tall structure standing upright"),
    ("furnishing", "artifact", "accessories that make a room or place functional"),
    ("furniture", "furnishing", "furnishings that make a room livable"),
    ("fixture", "furnishing", "an object firmly fixed in place"),
    ("living_thing", "entity", "a living or once-living entity"),
    ("organism", "living_thing", "a living thing that can function independently"),
    ("plant", "organism", "a living organism of the kind exemplified by trees"),
    ("tree", "plant", "a tall perennial woody plant with a main trunk"),
    ("shrub", "plant", "a low woody perennial plant with several stems"),
    ("herb", "plant", "a plant lacking a permanent woody stem"),
    ("flower", "plant", "a plant cultivated for its blossoms"),
    ("animal", "organism", "a living organism characterized by voluntary movement"),
    ("vertebrate", "animal", "an animal having a bony or cartilaginous skeleton"),
    ("mammal", "vertebrate", "a warm-blooded vertebrate that nourishes young with milk"),
    ("dog", "mammal", "a domesticated carnivorous mammal of the family Canidae"),
    ("cat", "mammal", "a small domesticated feline mammal kept as a pet"),
    ("horse", "mammal", "a large hoofed mammal domesticated for riding"),
    ("whale", "mammal", "a very large marine mammal of the order Cetacea"),
    ("bird", "vertebrate", "a warm-blooded egg-laying vertebrate with feathers"),
    ("eagle", "bird", "a large keen-sighted diurnal bird of prey"),
    ("sparrow", "bird", "a small brownish songbird of the family Passeridae"),
    ("penguin", "bird", "a flightless seabird of the southern hemisphere"),
    ("fish", "vertebrate", "any of various mostly cold-blooded aquatic vertebrates"),
    ("salmon", "fish", "a northern food fish that migrates upstream to spawn"),
    ("shark", "fish", "any of numerous elongate marine fishes with cartilage skeleton"),
    ("invertebrate", "animal", "any animal lacking a backbone or spinal column"),
    ("insect", "invertebrate", "small air-breathing arthropod with six legs"),
    ("worm", "invertebrate", "any of numerous elongate soft-bodied invertebrates"),
    ("mollusk", "invertebrate", "an invertebrate having a soft unsegmented body"),
    ("abstraction", "entity", "a general concept formed by extraction from instances"),
    ("attribute", "abstraction", "an abstraction belonging to a feature of something"),
    ("color", "attribute", "a visual attribute of things resulting from light reflection"),
    ("size", "attribute", "the physical magnitude of something"),
    ("weight", "attribute", "the vertical force exerted by a mass"),
    ("relation", "abstraction", "an abstraction belonging to two or more entities"),
    ("kinship", "relation", "family relation by blood or marriage"),
    ("ownership", "relation", "the relation of an owner to that which is possessed"),
    ("quantity", "abstraction", "how much there is of something"),
    ("number", "quantity", "a concept of quantity involving counting"),
    ("amount", "quantity", "the total volume or magnitude of something"),
    ("communication", "abstraction", "something that is communicated by words or signs"),
    ("language", "communication", "a systematic means of communicating by words"),
    ("writing", "communication", "letters or symbols used as a means of communication"),
    ("word", "language", "a unit of language that native speakers can identify"),
    ("sentence", "language", "a string of words satisfying grammatical rules"),
]

ROOT = "entity"


def build(seed=42, query_ratio=0.20, negsamples=5):
    rng = random.Random(seed)
    name2id = {n: i for i, (n, _, _) in enumerate(TAXO)}
    parent = {n: p for n, p, _ in TAXO}
    children = {}
    for n, p in parent.items():
        if p is not None:
            children.setdefault(p, []).append(n)

    def ancestors(n):
        out, cur = [], parent[n]
        while cur is not None:
            out.append(cur)
            cur = parent[cur]
        return out

    leaves = [n for n in parent
              if parent[n] is not None and n not in children]
    rng.shuffle(leaves)
    n_test = max(1, int(round(len(leaves) * query_ratio)))
    test_names = sorted(leaves[:n_test])
    train_names = [n for n in name2id if n not in test_names]

    test_ids = [name2id[n] for n in test_names]
    test_gt = [name2id[parent[n]] for n in test_names]

    def hard_cands(c):
        p = parent[c]
        cand = list(children.get(p, []))
        gp = parent[p]
        if gp is not None:
            uncles = [u for u in children.get(gp, []) if u != p]
            cand += uncles
            for u in uncles:
                cand += children.get(u, [])[:2]
        cand += [a for a in ancestors(c) if a != p][:2]
        return [x for x in dict.fromkeys(cand) if x != c and x != p]

    neg_dict, triples = {}, []
    for c in train_names:
        if parent[c] is None:
            continue
        cand = hard_cands(c)
        rng.shuffle(cand)
        picked = cand[:negsamples]
        if picked:
            neg_dict[name2id[c]] = [name2id[x] for x in picked]
            for x in picked:
                triples.append((name2id[c], name2id[parent[c]], name2id[x]))
    sibling_dict = {name2id[c]: [name2id[x] for x in children.get(parent[c], [])
                                 if x != c]
                    for c in train_names if parent[c] is not None}
    path2root = {name2id[n]: [name2id[a] for a in ancestors(n)]
                 for n in name2id}

    id2context = {name2id[n]: "%s : %s" % (n, g) for n, _, g in TAXO}
    train_child = [name2id[n] for n in train_names if parent[n] is not None]
    return dict(
        concept_set=set(name2id.values()),
        concept2id=name2id, id2concept={i: n for n, i in name2id.items()},
        id2context=id2context,
        train_concept_set=set(name2id[n] for n in train_names),
        train_parent_list=[name2id[parent[n]] for n in train_names
                            if parent[n] is not None],
        train_child_list=train_child,
        train_negative_parent_dict=neg_dict,
        train_sibling_dict=sibling_dict,
        child_parent_pair=[(name2id[n], name2id[parent[n]])
                           for n in parent if parent[n] is not None],
        child_neg_parent_pair=[(cid, nid) for cid, ns in neg_dict.items()
                               for nid in ns],
        child_sibling_pair=[(cid, s) for cid, ss in sibling_dict.items()
                            for s in ss],
        path2root=path2root,
        train_child_parent_negative_parent_triple=triples,
        test_concepts_id=test_ids, test_gt_id=test_gt,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        "data", "local", "taxonomy_data_local_.pkl"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    obj = build(seed=a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(obj, f)
    print("== 本地替代 pkl（官方 12 键 ID 三元组格式）==")
    print("  concepts=%d train=%d test=%d triples=%d neg_children=%d"
          % (len(obj["concept_set"]), len(obj["train_concept_set"]),
             len(obj["test_concepts_id"]),
             len(obj["train_child_parent_negative_parent_triple"]),
             len(obj["train_negative_parent_dict"])))
    print("  -> %s" % a.out)
    print("  诚实边界：真实概念名+人工定义的本地素材，非官方 5 数据集；"
          "结构/键名与官方一致，用于 --pkl 分支链路实证。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
