# Run (top-k accuracy, honest about ties):
#   python src/chess_eval/evaluation/eval_ranks.py \
#       --timing-ckpt ckpt_fix/timing/best.pt --timing-data src/chess_eval/graphs/test \
#       --plain-ckpt  ckpt_fix/plain/best.pt  --plain-data  src/chess_eval/graphs/test_plain \
#       --topk 1 3 5 --out risultati/fix/results_ranks.json
#
# Top-k accuracy that is honest about ties. 34% of test positions have several
# legal moves sharing the top logit: the network calls them equivalent, and
# argmax vs topk pick different ones (a 2-point swing that is not a property of
# the model). The node features carry no square identity, so two empty squares
# with the same edges are the same node and genuinely tie.
#
# So per example, over the legal moves only, it counts:
#     better = moves scoring STRICTLY higher than the true move
#     tied   = OTHER moves scoring EXACTLY the same
# and reports three numbers with no tie-breaking: strict (better+tied < k, lower
# bound), expected (prob. of landing in top k under a uniform random tie-break,
# computed exactly), optimistic (better < k, upper bound). The old 26.62% sits
# between strict and optimistic.
#
# McNemar needs a yes/no per puzzle, so it runs on the two bounds, not on the
# expected value: if timing beats plain under BOTH readings, the conclusion does
# not depend on how ties are resolved.

import argparse
import json
import math

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from chess_eval.common.graph_dataset import ParquetGraphDataset, legal_mask
from chess_eval.evaluation.evaluate_models import load_model, mcnemar


def rank_all(model, data_dir, with_time, device, batch_size=256):
    """One pass over the test set: per-example better/tied/depth/puzzle_id/n_legal.

    Records how many moves beat the true move and how many draw with it; every
    value of k is then arithmetic, so k is not needed here.
    """
    ds = ParquetGraphDataset(data_dir, with_time=with_time,
                             keep_meta=True, shuffle=False)
    loader = DataLoader(ds, batch_size=batch_size)

    better, tied, depth, pid, nlegal = [], [], [], [], []
    illegal_target = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch.puzzle_id
            batch = batch.to(device)
            logits = model(batch) + legal_mask(batch).to(device)
            true = logits.gather(1, batch.y.view(-1, 1))          # (B, 1)
            # the solution must itself be a legal move; if it were masked out
            # its logit would be -inf and every other masked move would "tie"
            # with it, quietly corrupting the counts
            illegal_target += int(torch.isinf(true).sum())
            better.extend((logits > true).sum(dim=1).cpu().tolist())
            tied.extend(((logits == true).sum(dim=1) - 1).cpu().tolist())
            depth.extend(batch.n_remaining.cpu().tolist())
            nlegal.extend(batch.n_legal.cpu().tolist())
            pid.extend(list(ids))

    if illegal_target:
        raise SystemExit(
            f"{illegal_target} esempi hanno la mossa soluzione fuori dalle "
            f"mosse legali: i conteggi sarebbero falsati")

    return {"better": np.array(better, dtype=np.int64),
            "tied": np.array(tied, dtype=np.int64),
            "depth": np.array(depth),
            "puzzle_id": np.array(pid, dtype=object),
            "n_legal": np.array(nlegal, dtype=float)}


def outcomes(better, tied, k):
    """Per-example (strict, expected, optimistic) for a given k.

    strict and optimistic are booleans, expected is a probability in [0, 1].
    See the header for the formulas.
    """
    optimistic = better < k
    strict = (better + tied) < k
    slots = np.clip(k - better, 0, tied + 1)
    expected = slots / (tied + 1.0)
    return strict, expected, optimistic


def baseline(n_legal, k):
    """Accuracy of guessing k distinct legal moves at random: mean of k/n (1 if k>=n)."""
    return float(np.mean(np.minimum(k, n_legal) / n_legal))


def ci95(p, n):
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n) if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing-ckpt", required=True)
    ap.add_argument("--timing-data", required=True)
    ap.add_argument("--plain-ckpt", required=True)
    ap.add_argument("--plain-data", required=True)
    ap.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="results_ranks.json")
    args = ap.parse_args()

    ks = sorted(set(args.topk))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\ntop-k: {ks}")

    res = {}
    for name, ck, data, with_time in (
            ("timing", args.timing_ckpt, args.timing_data, True),
            ("plain", args.plain_ckpt, args.plain_data, False)):
        model, a, epoch, best_val = load_model(ck, device)
        assert a["timing"] == with_time, \
            f"{name}: il checkpoint ha timing={a['timing']}"
        print(f"\n{name}: epoca {epoch}, val loss {best_val:.4f}")
        res[name] = rank_all(model, data, with_time, device, args.batch_size)
        print(f"  {len(res[name]['better']):,} puzzle")

    t, pl = res["timing"], res["plain"]
    assert np.array_equal(t["puzzle_id"], pl["puzzle_id"]), \
        "i due test set non sono allineati"
    print("\ngli stessi puzzle nello stesso ordine: True")

    # ---- quanto pesano i pareggi ----
    print(f"\n{'=' * 78}\nSTRUTTURA DEI PAREGGI\n{'=' * 78}")
    print(f"{'modello':>8} {'mosse/pos':>11} {'best unico':>12} "
          f"{'pareggi al top':>16} {'media pari merito':>19}")
    report = {"k": ks, "ties": {}, "topk": {}}
    for name in ("timing", "plain"):
        r = res[name]
        at_top = r["better"] == 0                    # true move is a maximum
        unique = at_top & (r["tied"] == 0)
        # how many moves share the maximum, over the examples where we can see
        # it (the true move is one of them)
        avg_tied = float(r["tied"][at_top].mean()) + 1 if at_top.any() else 0.0
        row = {"legal_mean": float(r["n_legal"].mean()),
               "at_top": float(at_top.mean()),
               "unique_top": float(unique.mean()),
               "avg_tied_at_top": avg_tied}
        report["ties"][name] = row
        print(f"{name:>8} {row['legal_mean']:>11.1f} "
              f"{row['unique_top']:>11.2%} {row['at_top'] - row['unique_top']:>15.2%}"
              f" {avg_tied:>19.2f}")
    print("\n'best unico' = la soluzione e' l'unica mossa col punteggio massimo.")
    print("'pareggi al top' = la soluzione e' fra i massimi, ma a pari merito.")

    # ---- accuratezza ----
    for k in ks:
        base = baseline(t["n_legal"], k)
        print(f"\n{'=' * 78}\nTOP-{k}   (baseline caso: {base:.2%})\n{'=' * 78}")
        print(f"{'n':>4} {'puzzle':>9} | {'timing: min':>12} {'atteso':>9} "
              f"{'max':>9} | {'plain: min':>12} {'atteso':>9} {'max':>9}")

        k_rep = {}
        for d in list(sorted(set(t["depth"].tolist()))) + ["tot"]:
            m = (np.ones_like(t["depth"], dtype=bool) if d == "tot"
                 else t["depth"] == d)
            cells, row = [], {"puzzles": int(m.sum())}
            for name in ("timing", "plain"):
                r = res[name]
                s, e, o = outcomes(r["better"][m], r["tied"][m], k)
                row[name] = {"strict": float(s.mean()),
                             "expected": float(e.mean()),
                             "optimistic": float(o.mean()),
                             "ci": ci95(float(e.mean()), int(m.sum()))}
                cells.append(f"{s.mean():>11.2%} {e.mean():>9.2%} "
                             f"{o.mean():>9.2%}")
            row["baseline"] = (base if d == "tot"
                               else baseline(t["n_legal"][m], k))
            k_rep[str(d)] = row
            label = "tot" if d == "tot" else f"{d}"
            print(f"{label:>4} {row['puzzles']:>9,} | {cells[0]} | {cells[1]}")

        # McNemar sui due estremi: se regge in entrambi, la conclusione non
        # dipende da come si rompono i pareggi
        st, et, ot = outcomes(t["better"], t["tied"], k)
        sp, ep, op = outcomes(pl["better"], pl["tied"], k)
        mc_s, mc_o = mcnemar(st, sp), mcnemar(ot, op)
        k_rep["mcnemar"] = {"strict": mc_s, "optimistic": mc_o}
        print(f"\n  differenza timing - plain, totale:")
        print(f"    pessimistica {st.mean() - sp.mean():+.2%}   "
              f"b={mc_s['b']:,} c={mc_s['c']:,} p={mc_s['p']:.2}"
              f"{'  *' if mc_s['p'] < 0.05 else ''}")
        print(f"    attesa       {et.mean() - ep.mean():+.2%}")
        print(f"    ottimistica  {ot.mean() - op.mean():+.2%}   "
              f"b={mc_o['b']:,} c={mc_o['c']:,} p={mc_o['p']:.2}"
              f"{'  *' if mc_o['p'] < 0.05 else ''}")
        report["topk"][f"top{k}"] = k_rep

    print(f"\n{'=' * 78}\nRIEPILOGO (totale)\n{'=' * 78}")
    print(f"{'k':>4} {'baseline':>10} | {'timing atteso':>14} "
          f"{'plain atteso':>13} {'diff':>8} | {'vecchio (argmax)':>17}")
    for k in ks:
        r = report["topk"][f"top{k}"]["tot"]
        print(f"{k:>4} {r['baseline']:>10.2%} | {r['timing']['expected']:>14.2%} "
              f"{r['plain']['expected']:>13.2%} "
              f"{r['timing']['expected'] - r['plain']['expected']:>+8.2%} |"
              f"{'  26.62% / 25.72%' if k == 1 else '':>17}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(f"\nrisultati salvati in {args.out}")


if __name__ == "__main__":
    main()