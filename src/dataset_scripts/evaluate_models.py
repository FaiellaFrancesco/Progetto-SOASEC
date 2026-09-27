"""
evaluate_models.py

The final measurement: the two trained models on the TEST set, which until now
has never been touched.

    python evaluate_models.py \\
        --timing-ckpt ckpt/timing/best.pt --timing-data graphs/test \\
        --plain-ckpt  ckpt/plain/best.pt  --plain-data  graphs/test_plain \\
        --out results_test.json

WHY TWO DATA FOLDERS
--------------------
The two models take a different number of node features: 13 with timing (the
think_time column) and 12 without. graphs/test was built with --timing, so it
feeds the timing model; graphs/test_plain is the same rows built without it.
Both come from the same test.csv in the same order, and the script CHECKS that
the puzzles line up before comparing anything.

WHY McNEMAR AND NOT TWO CONFIDENCE INTERVALS
---------------------------------------------
The two models answer the SAME puzzles, so their answers are correlated.
Comparing two independent proportions throws that away and loses power.
McNemar looks only at the puzzles where the two disagree:

    b = timing right, plain wrong
    c = plain right, timing wrong

Under the null hypothesis b and c come from the same coin. The puzzles both got
right, or both got wrong, carry no information about which model is better and
are correctly ignored.

WHAT THE NUMBERS MEAN
---------------------
This is the test set: these are the numbers for the report. They are measured
once. Do not tune anything on them - the moment you go back and change a
hyperparameter because of what you saw here, the test set stops being a test
set and becomes a second validation set.
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from graph_dataset import ParquetGraphDataset, legal_mask
from model import ChessMoveGNN


"""
INPUT: the path of a checkpoint written by train.py, and a device
OUTPUT: (model in eval mode, the args it was trained with)

The hyperparameters travel inside the checkpoint, so the architecture is
rebuilt exactly as it was trained. Guessing them here would silently produce a
different network and load the weights into the wrong shapes.
"""


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    n_features = 13 if a["timing"] else 12
    model = ChessMoveGNN(
        num_event_features=n_features,
        num_layers=a["num_layers"],
        lambda_decay=a["lambda_decay"],
        gat_hidden_dim_event=a["hidden"],
        gat_hidden_dim_embed=a["hidden"],
        gat_hidden_dim_concat=a["hidden"],
        num_heads=a["heads"],
        dropout=a["dropout"],
        attention_softmax=a.get("attention_softmax", True),
        edge_inputs=a.get("edge_inputs", False),
        self_loops=a.get("self_loops", False),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, a, ck.get("epoch"), ck.get("best_val")


"""
INPUT: a model, the folder of test parquet, timing flag, device, batch size
OUTPUT: dict with per-example arrays: correct, depth, puzzle_id, n_legal

Runs the whole test set once. Returns per-example results rather than a summary
because McNemar needs to know WHICH puzzles each model got right, not just how
many.
"""


def predict_all(model, data_dir, with_time, device, batch_size=256):
    ds = ParquetGraphDataset(data_dir, with_time=with_time,
                             keep_meta=True, shuffle=False)
    loader = DataLoader(ds, batch_size=batch_size)

    correct, depth, pid, nlegal = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch.puzzle_id            # collated into a list
            batch = batch.to(device)
            mask = legal_mask(batch).to(device)
            pred = (model(batch) + mask).argmax(dim=1)
            correct.extend((pred == batch.y).cpu().tolist())
            depth.extend(batch.n_remaining.cpu().tolist())
            nlegal.extend(batch.n_legal.cpu().tolist())
            pid.extend(list(ids))
    return {"correct": np.array(correct, dtype=bool),
            "depth": np.array(depth),
            "puzzle_id": np.array(pid, dtype=object),
            "n_legal": np.array(nlegal, dtype=float)}


"""
INPUT: the per-example correctness and depth arrays
OUTPUT: {n: {"n": count, "acc": .., "ci": ..}} plus "overall"

The confidence interval is the normal approximation on a proportion. On n=5,
where the test set has few puzzles, it is wide on purpose: it is the number
that stops you from claiming a difference the data cannot support.
"""


def summarise(res, baseline_by_depth=None):
    out = {}
    for d in sorted(set(res["depth"].tolist())):
        m = res["depth"] == d
        k, n = int(res["correct"][m].sum()), int(m.sum())
        a = k / n if n else 0.0
        ci = 1.96 * math.sqrt(a * (1 - a) / n) if n else 0.0
        row = {"n": n, "correct": k, "acc": a, "ci": ci}
        if baseline_by_depth is not None:
            row["baseline"] = baseline_by_depth.get(d)
        out[int(d)] = row
    k, n = int(res["correct"].sum()), len(res["correct"])
    a = k / n
    out["overall"] = {"n": n, "correct": k, "acc": a,
                      "ci": 1.96 * math.sqrt(a * (1 - a) / n)}
    return out


"""
INPUT: two boolean arrays of correctness, aligned example by example
OUTPUT: dict with b, c, the chi-square statistic and the p-value

McNemar with continuity correction. b and c are the disagreements; examples
where the two models agree are ignored because they say nothing about which is
better.
"""


def mcnemar(a_correct, b_correct):
    b = int(np.sum(a_correct & ~b_correct))   # first right, second wrong
    c = int(np.sum(~a_correct & b_correct))   # second right, first wrong
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / n
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, 1))
    except ImportError:
        # survival function of chi-square with 1 dof = erfc(sqrt(x/2))
        p = math.erfc(math.sqrt(chi2 / 2))
    return {"b": b, "c": c, "chi2": chi2, "p": p}


def print_table(title, s):
    print(f"\n{title}")
    print(f"{'n':>3} {'puzzle':>9} {'giusti':>9} {'accuratezza':>13} {'+/- 95%':>9}"
          f" {'baseline':>10} {'volte':>7}")
    for d in sorted(k for k in s if k != "overall"):
        r = s[d]
        base = r.get("baseline")
        bs = f"{base:>9.2%}" if base else " " * 10
        mult = f"{r['acc']/base:>6.1f}x" if base else " " * 7
        print(f"{d:>3} {r['n']:>9,} {r['correct']:>9,} {r['acc']:>12.2%} "
              f"{r['ci']:>8.2%} {bs} {mult}")
    o = s["overall"]
    print(f"{'tot':>3} {o['n']:>9,} {o['correct']:>9,} {o['acc']:>12.2%} "
          f"{o['ci']:>8.2%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timing-ckpt", required=True)
    p.add_argument("--timing-data", required=True)
    p.add_argument("--plain-ckpt", required=True)
    p.add_argument("--plain-data", required=True)
    p.add_argument("--out", default="results_test.json")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    results = {}
    per_example = {}
    for name, ck_path, data_dir, with_time in (
            ("timing", args.timing_ckpt, args.timing_data, True),
            ("plain", args.plain_ckpt, args.plain_data, False)):
        model, a, epoch, best_val = load_model(ck_path, device)
        print(f"\n{name}: epoca {epoch}, miglior val loss {best_val:.4f}, "
              f"layers {a['num_layers']}, timing {a['timing']}")
        # a checkpoint trained with --timing must not be scored without it,
        # and vice versa: the feature count would not even match
        assert a["timing"] == with_time, (
            f"{name}: il checkpoint ha timing={a['timing']} ma lo valuti con "
            f"with_time={with_time}")
        per_example[name] = predict_all(model, data_dir, with_time, device,
                                        args.batch_size)
        print(f"  valutati {len(per_example[name]['correct']):,} puzzle")

    t, pl = per_example["timing"], per_example["plain"]

    # the two runs must be the same puzzles in the same order, or every
    # comparison below is meaningless
    assert len(t["correct"]) == len(pl["correct"]), \
        "i due test set hanno un numero diverso di esempi"
    same = np.array_equal(t["puzzle_id"], pl["puzzle_id"])
    print(f"\ngli stessi puzzle nello stesso ordine: {same}")
    assert same, "i due test set non sono allineati: non si possono confrontare"

    # baseline: picking uniformly among the legal moves is 1/n_legal, averaged
    # over the puzzles of that depth - not 1 over the average, which differs
    base = {}
    for d in sorted(set(t["depth"].tolist())):
        m = t["depth"] == d
        base[int(d)] = float(np.mean(1.0 / t["n_legal"][m]))

    results["timing"] = summarise(t, base)
    results["plain"] = summarise(pl, base)

    print_table("MODELLO CON IL TEMPO - test set", results["timing"])
    print_table("MODELLO SENZA IL TEMPO - test set", results["plain"])

    # ---- confronto ----
    print("\nCONFRONTO (McNemar: guarda solo i puzzle su cui i due discordano)")
    print(f"{'n':>3} {'plain':>9} {'timing':>9} {'diff':>9} {'b':>7} {'c':>7} "
          f"{'p':>10} {'':>6}")
    results["mcnemar"] = {}
    for d in sorted(k for k in results["timing"] if k != "overall"):
        m = t["depth"] == d
        mc = mcnemar(t["correct"][m], pl["correct"][m])
        results["mcnemar"][int(d)] = mc
        at, ap = results["timing"][d]["acc"], results["plain"][d]["acc"]
        star = "*" if mc["p"] < 0.05 else ""
        print(f"{d:>3} {ap:>8.2%} {at:>9.2%} {at-ap:>+9.2%} {mc['b']:>7,} "
              f"{mc['c']:>7,} {mc['p']:>10.2}  {star}")
    mc = mcnemar(t["correct"], pl["correct"])
    results["mcnemar"]["overall"] = mc
    at = results["timing"]["overall"]["acc"]
    ap = results["plain"]["overall"]["acc"]
    star = "*" if mc["p"] < 0.05 else ""
    print(f"{'tot':>3} {ap:>8.2%} {at:>9.2%} {at-ap:>+9.2%} {mc['b']:>7,} "
          f"{mc['c']:>7,} {mc['p']:>10.2}  {star}")
    print("\n* = differenza significativa al 5%.  "
          "b = solo timing giusto, c = solo plain giusto")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"\nrisultati salvati in {args.out}")


if __name__ == "__main__":
    main()