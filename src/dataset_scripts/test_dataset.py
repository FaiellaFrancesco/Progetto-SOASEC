"""
tests_dataset.py

Checks graph_dataset.py: the parquet -> PyG conversion, the layout of
edge_attr, the batching, and a real forward pass through a GATConv.

    python tests_dataset.py -i ./prova_out          (built with --timing)

tests.py covers the encoder, verify_pipeline covers the parquet, this covers
the handover to the model.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

from graph_dataset import (ParquetGraphDataset, to_pyg_data, legal_mask,
                           N_EDGE_TYPES)

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
    n_feat = len(np.asarray(row["x"].tolist()))
    check("x is (64, F)", d.x.shape[0] == 64, tuple(d.x.shape))
    check("x is float32", d.x.dtype == torch.float32)
    check("edge_index has 2 rows", d.edge_index.shape[0] == 2)
    check("edge_index is int64", d.edge_index.dtype == torch.int64)
    check("square indices stay in 0..63",
          int(d.edge_index.min()) >= 0 and int(d.edge_index.max()) <= 63)
    check("edge_attr has 1 + 4 columns",
          d.edge_attr.shape[1] == 1 + N_EDGE_TYPES, tuple(d.edge_attr.shape))
    check("one row of edge_attr per edge",
          d.edge_attr.shape[0] == d.edge_index.shape[1])

    # the whole point of the file: column 0 must be the time, not a one-hot
    t = torch.tensor(np.asarray(row["edge_time"], dtype=np.float32))
    check("edge_attr[:, 0] IS delta_t",
          torch.allclose(d.edge_attr[:, 0], t))
    check("the 4 type columns still one-hot after the shift",
          torch.all(d.edge_attr[:, 1:].sum(dim=1) == 1.0))
    check("column 0 is NOT a one-hot (it would mean the layout is wrong)",
          not torch.all((d.edge_attr[:, 0] == 0) | (d.edge_attr[:, 0] == 1)))

    check("y is a 1-element long tensor",
          d.y.shape == (1,) and d.y.dtype == torch.int64)
    check("the label is among the legal moves", int(d.y) in d.legal_moves)
    check("n_legal matches legal_moves", int(d.n_legal) == len(d.legal_moves))
    check("metadata is off by default", not hasattr(d, "fen"))
    check("keep_meta=True carries the fen",
          isinstance(to_pyg_data(row, keep_meta=True).fen, str))

    # ---------------- time_divisor ----------------
    print("\n--- time_divisor ---")
    d20 = to_pyg_data(row, with_time=True, time_divisor=20.0)
    check("dividing by 20 scales delta_t",
          torch.allclose(d20.edge_attr[:, 0], t / 20.0))
    check("dividing does not touch the type columns",
          torch.allclose(d20.edge_attr[:, 1:], d.edge_attr[:, 1:]))

    # ---------------- to_pyg_data, without time ----------------
    print("\n--- to_pyg_data (with_time=False) ---")
    d0 = to_pyg_data(row, with_time=False)
    check("edge_attr has 4 columns", d0.edge_attr.shape[1] == N_EDGE_TYPES,
          tuple(d0.edge_attr.shape))
    check("it is exactly the one-hot the encoder produced",
          torch.allclose(d0.edge_attr, d.edge_attr[:, 1:]))

    # ---------------- batching ----------------
    print("\n--- batching ---")
    ds = ParquetGraphDataset(args.input_dir, with_time=True, shuffle=False)
    loader = DataLoader(ds, batch_size=8)
    batch = next(iter(loader))
    check("the batch holds 8 graphs", int(batch.num_graphs) == 8)
    check("x is stacked to 8 * 64 rows", batch.x.shape[0] == 8 * 64,
          batch.x.shape[0])
    check("y is one label per graph", batch.y.shape == (8,))
    check("edge_index was offset per graph, so it reaches past 63",
          int(batch.edge_index.max()) > 63, int(batch.edge_index.max()))
    check("legal_moves was NOT offset (they are move classes, not nodes)",
          int(batch.legal_moves.max()) < 4096, int(batch.legal_moves.max()))
    check("n_legal sums to the length of legal_moves",
          int(batch.n_legal.sum()) == batch.legal_moves.shape[0])

    # ---------------- legal_mask ----------------
    print("\n--- legal_mask ---")
    m = legal_mask(batch)
    check("mask is (8, 4096)", m.shape == (8, 4096), tuple(m.shape))
    check("exactly n_legal zeros per row",
          all(int((m[i] == 0).sum()) == int(batch.n_legal[i])
              for i in range(8)))
    check("every true label is unmasked",
          all(m[i, batch.y[i]] == 0.0 for i in range(8)))
    logits = torch.randn(8, 4096)
    picked = (logits + m).argmax(dim=1)
    check("argmax on masked logits always lands on a legal move",
          all(int(picked[i]) in batch.legal_moves[
              int(batch.n_legal[:i].sum()):
              int(batch.n_legal[:i + 1].sum())] for i in range(8)))

    # ---------------- a real forward pass ----------------
    print("\n--- forward pass through GATConv ---")
    conv = GATConv(batch.x.shape[1], 32, heads=4, edge_dim=batch.edge_attr.shape[1])
    out = conv(batch.x, batch.edge_index, batch.edge_attr)
    check("GATConv accepts the batch", out.shape == (8 * 64, 32 * 4),
          tuple(out.shape))
    check("the output has no NaN", not bool(torch.isnan(out).any()))

    # the decay TimeGNN applies, computed on our real delta_t
    for lam in (0.01, 0.05):
        decay = torch.exp(-lam * batch.edge_attr[:, 0])
        check(f"lambda={lam}: decay stays in (0, 1]",
              bool((decay > 0).all() and (decay <= 1).all()),
              f"min {float(decay.min()):.3f}")

    # ---------------- streaming ----------------
    print("\n--- streaming ---")
    n = sum(1 for _ in ParquetGraphDataset(args.input_dir, shuffle=False))
    total = sum(len(pd.read_parquet(f)) for f in files)
    check("the dataset yields every row of every shard", n == total,
          f"{n} vs {total}")
    a = [int(d.y) for d in ParquetGraphDataset(args.input_dir, shuffle=False)][:50]
    s1 = ParquetGraphDataset(args.input_dir, shuffle=True, seed=1)
    b = [int(d.y) for d in s1][:50]
    check("shuffle=True changes the order", a != b)
    s2 = ParquetGraphDataset(args.input_dir, shuffle=True, seed=1)
    c = [int(d.y) for d in s2][:50]
    check("the same seed gives the same order", b == c)
    s1.set_epoch(1)
    d1 = [int(d.y) for d in s1][:50]
    check("set_epoch changes the order", b != d1)

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    if all(RESULTS):
        print("the adapter is ready for the model")
        return 0
    print("something is broken, look at the FAIL lines above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
