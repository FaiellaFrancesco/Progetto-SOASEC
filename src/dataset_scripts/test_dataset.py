"""
test_dataset.py

Checks graph_dataset.py against the contract DualGATTimeAwareETModel actually
has, read off the library's source: event_ids, edge_type as indices, time as
its own field. Ends with a real forward and backward pass through the library's
model, so a mismatch shows up here and not after four hours on Colab.

    python test_dataset.py -i graphs/train_timing

The model checks are skipped with a clear message if timegnn is not installed,
so this still runs on a machine that only has the preprocessing side.
"""

import argparse
import glob
import os
import sys
from itertools import islice

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from graph_dataset import (ParquetGraphDataset, to_pyg_data, legal_mask,
                           piece_ids_from_x, N_EDGE_TYPES, N_PIECE_IDS)

RESULTS = []


def check(name, condition, extra=""):
    RESULTS.append(bool(condition))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}"
          f"{'  ' + str(extra) if extra != '' else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True,
                        help="a folder of parquet built with --timing")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    df = pd.read_parquet(files[0])
    row = df.iloc[0]

    # ---------------- to_pyg_data, with time ----------------
    print("\n--- to_pyg_data (with_time=True) ---")
    d = to_pyg_data(row, with_time=True)
    check("x is (64, F)", d.x.shape[0] == 64, tuple(d.x.shape))
    check("x is float32", d.x.dtype == torch.float32)
    check("edge_index has 2 rows", d.edge_index.shape[0] == 2)
    check("square indices stay in 0..63",
          int(d.edge_index.min()) >= 0 and int(d.edge_index.max()) <= 63)

    # the three fields the library reads, in the shapes it needs
    check("time is a 1-D float tensor, one value per edge",
          d.time.dim() == 1 and d.time.shape[0] == d.edge_index.shape[1]
          and d.time.dtype == torch.float32, tuple(d.time.shape))
    check("time carries the real delta_t",
          torch.allclose(d.time,
                         torch.tensor(np.asarray(row["edge_time"],
                                                 dtype=np.float32))))
    check("edge_type is an int index, not a one-hot",
          d.edge_type.dim() == 1 and d.edge_type.dtype == torch.int64,
          tuple(d.edge_type.shape))
    check("edge_type stays inside the embedding table",
          int(d.edge_type.min()) >= 0 and int(d.edge_type.max()) < N_EDGE_TYPES)
    check("edge_type matches the argmax of the stored one-hot",
          torch.equal(d.edge_type,
                      torch.tensor(np.argmax(
                          np.asarray(row["edge_attr"].tolist()), axis=1))))
    check("event_ids is one int per square",
          d.event_ids.shape == (64,) and d.event_ids.dtype == torch.int64)
    check("event_ids stays inside the embedding table",
          int(d.event_ids.min()) >= 0 and int(d.event_ids.max()) < N_PIECE_IDS)
    check("no edge_attr: the model builds it from edge_type",
          "edge_attr" not in d)

    check("y is a 1-element long tensor",
          d.y.shape == (1,) and d.y.dtype == torch.int64)
    check("the label is among the legal moves", int(d.y) in d.legal_moves)
    check("n_legal matches legal_moves", int(d.n_legal) == len(d.legal_moves))
    check("metadata is off by default", not hasattr(d, "fen"))
    check("keep_meta=True carries the fen",
          isinstance(to_pyg_data(row, keep_meta=True).fen, str))

    # ---------------- piece_ids_from_x ----------------
    print("\n--- piece_ids_from_x ---")
    x = np.asarray(row["x"].tolist(), dtype=np.float32)
    ids = piece_ids_from_x(x)
    empty = x[:, 8] > 0.5
    check("empty squares get id 0", bool((ids[empty] == 0).all()))
    check("occupied squares never get 0", bool((ids[~empty] != 0).all()))
    check("number of pieces matches the empty flag",
          int((ids != 0).sum()) == int((~empty).sum()),
          f"{int((ids != 0).sum())} pezzi")
    check("white ids are 1..6, black 7..12",
          bool(((ids[(x[:, 6] > .5)] >= 1) & (ids[(x[:, 6] > .5)] <= 6)).all()
               and ((ids[(x[:, 7] > .5)] >= 7)
                    & (ids[(x[:, 7] > .5)] <= 12)).all()))

    # ---------------- time_divisor and with_time=False ----------------
    print("\n--- time scaling and the ablation switch ---")
    d20 = to_pyg_data(row, with_time=True, time_divisor=20.0)
    check("time_divisor scales delta_t",
          torch.allclose(d20.time, d.time / 20.0))
    d0 = to_pyg_data(row, with_time=False)
    check("with_time=False gives all-zero time",
          bool((d0.time == 0).all()) and d0.time.shape == d.time.shape)
    check("switching time off leaves the graph identical",
          torch.equal(d0.edge_index, d.edge_index)
          and torch.equal(d0.edge_type, d.edge_type))
    check("zero time means a decay of exactly 1 on every edge",
          bool(torch.allclose(torch.exp(-0.05 * d0.time),
                              torch.ones_like(d0.time))))

    # ---------------- batching ----------------
    print("\n--- batching ---")
    ds = ParquetGraphDataset(args.input_dir, with_time=True, shuffle=False)
    loader = DataLoader(ds, batch_size=8)
    batch = next(iter(loader))
    check("the batch holds 8 graphs", int(batch.num_graphs) == 8)
    check("x is stacked to 8 * 64 rows", batch.x.shape[0] == 8 * 64)
    check("event_ids is stacked the same way", batch.event_ids.shape[0] == 8 * 64)
    check("y is one label per graph", batch.y.shape == (8,))
    check("edge_index was offset per graph, so it reaches past 63",
          int(batch.edge_index.max()) > 63, int(batch.edge_index.max()))
    check("edge_type was NOT offset (it indexes a 4-row embedding)",
          int(batch.edge_type.max()) < N_EDGE_TYPES, int(batch.edge_type.max()))
    check("event_ids was NOT offset (it indexes a 13-row embedding)",
          int(batch.event_ids.max()) < N_PIECE_IDS, int(batch.event_ids.max()))
    check("legal_moves was NOT offset (move classes, not node indices)",
          int(batch.legal_moves.max()) < 4096)
    check("time has one value per edge of the whole batch",
          batch.time.shape[0] == batch.edge_index.shape[1])
    check("n_legal sums to the length of legal_moves",
          int(batch.n_legal.sum()) == batch.legal_moves.shape[0])

    # ---------------- legal_mask ----------------
    print("\n--- legal_mask ---")
    m = legal_mask(batch)
    check("mask is (8, 4096)", m.shape == (8, 4096))
    check("exactly n_legal zeros per row",
          all(int((m[i] == 0).sum()) == int(batch.n_legal[i])
              for i in range(8)))
    check("every true label is unmasked",
          all(m[i, batch.y[i]] == 0.0 for i in range(8)))
    torch.manual_seed(0)
    logits = torch.randn(8, 4096)
    picked = (logits + m).argmax(dim=1)
    off = 0
    ok = True
    for i, k in enumerate(batch.n_legal.tolist()):
        ok = ok and int(picked[i]) in batch.legal_moves[off:off + k]
        off += k
    check("argmax on masked logits always lands on a legal move", ok)

    # ---------------- the real model ----------------
    print("\n--- forward through the library's own model ---")
    try:
        from timegnn.models.gat_time_decay_status_emb import DualGATTimeAwareETModel
    except ImportError as e:
        # do NOT say "not installed": ModuleNotFoundError is a subclass of
        # ImportError, so a missing dependency of timegnn's own __init__
        # (seaborn, matplotlib) lands here too and the real cause matters.
        print(f"[SKIP] impossibile importare il modello -> {type(e).__name__}: {e}")
        print("       se manca seaborn/matplotlib: pip install seaborn matplotlib")
        DualGATTimeAwareETModel = None

    if DualGATTimeAwareETModel is not None:
        model = DualGATTimeAwareETModel(
            num_event_features=batch.x.shape[1],
            num_embedding_features=N_PIECE_IDS,
            embedding_dims=64,
            gat_hidden_dim_event=32, gat_hidden_dim_embed=32,
            gat_hidden_dim_concat=64,
            output_dim=64,                 # per node; the head comes after
            num_heads=4, num_edge_types=N_EDGE_TYPES, edge_type_dim=16,
            lambda_decay=0.05, num_layers=3, dropout=0.1)
        out = model(batch)
        check("the model accepts our batch unchanged",
              out.shape == (8 * 64, 64), tuple(out.shape))
        check("the output has no NaN", not bool(torch.isnan(out).any()))

        B = int(batch.num_graphs)
        h = out.view(B, 64, -1)
        Wf = torch.nn.Linear(h.shape[-1], 64, bias=False)
        Wt = torch.nn.Linear(h.shape[-1], 64, bias=False)
        pol = torch.bmm(Wf(h), Wt(h).transpose(1, 2)).reshape(B, 4096)
        check("the bilinear head gives 4096 logits per graph",
              pol.shape == (8, 4096))
        loss = torch.nn.functional.cross_entropy(pol + m, batch.y)
        loss.backward()
        expected = np.log(float(batch.n_legal.float().mean()))
        check("loss is finite and backward runs", torch.isfinite(loss),
              f"loss={loss.item():.3f}")
        check("no NaN in the gradients",
              all(not bool(torch.isnan(p.grad).any())
                  for p in model.parameters() if p.grad is not None))
        # An untrained net should be indistinguishable from guessing among the
        # legal moves, so the loss should sit near ln(mean legal moves). It can
        # start higher: TimeAwareETGATConv does NOT normalise attention (the
        # softmax is commented out in the library), so activations can grow
        # across layers and the net starts over-confident. Informational, not
        # a hard failure - but a loss far above this means the first epochs
        # will mostly be spent unlearning the initialisation.
        ratio = loss.item() / expected
        check("loss is in the same ballpark as guessing among legal moves",
              torch.isfinite(loss) and loss.item() < 10 * expected,
              f"loss={loss.item():.2f}  atteso~{expected:.2f}  "
              f"rapporto={ratio:.1f}x  "
              # only the legal logits reach the loss: the rest are -inf
              f"logit legali |max|={pol[m == 0].abs().max().item():.1f}")

    # ---------------- streaming ----------------
    # NOTE: these run on ONE shard, not the whole folder. The real training set
    # has ~470k examples and iterating it four times here would take minutes for
    # no extra information: the streaming logic is per-file anyway.
    print("\n--- streaming (un solo shard) ---")
    shard = files[0]
    n = sum(1 for _ in ParquetGraphDataset(shard, shuffle=False))
    total = len(pd.read_parquet(shard))
    check("the dataset yields every row of the shard", n == total,
          f"{n} vs {total}")

    def first_labels(ds, k=50):
        # islice, NOT [:k] on a list comprehension: the comprehension would
        # materialise every example first
        return [int(x.y) for x in islice(iter(ds), k)]

    a = first_labels(ParquetGraphDataset(shard, shuffle=False))
    s1 = ParquetGraphDataset(shard, shuffle=True, seed=1)
    b = first_labels(s1)
    check("shuffle=True changes the order", a != b)
    s2 = ParquetGraphDataset(shard, shuffle=True, seed=1)
    check("the same seed gives the same order", b == first_labels(s2))
    s1.set_epoch(1)
    check("set_epoch changes the order", b != first_labels(s1))

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    if all(RESULTS):
        print("l'adattatore parla la lingua del modello")
        return 0
    print("something is broken, look at the FAIL lines above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
