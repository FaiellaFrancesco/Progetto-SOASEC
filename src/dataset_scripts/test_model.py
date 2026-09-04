"""
test_model.py

Checks model.py: the bilinear head's indexing, and the numerical stability the
attention fix buys.

    python test_model.py -i graphs/train_timing

The stability table is the reason model.py exists, so it is a test and not a
comment: if a future change to the library or to the fix breaks it, this fails.
"""

import argparse
import statistics
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from graph_dataset import ParquetGraphDataset, legal_mask
from model import ChessMoveGNN, normalise_attention, N_CLASSES

RESULTS = []


def check(name, condition, extra=""):
    RESULTS.append(bool(condition))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}"
          f"{'  ' + str(extra) if extra != '' else ''}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input_dir", required=True)
    p.add_argument("--seeds", type=int, default=6)
    args = p.parse_args()

    ds = ParquetGraphDataset(args.input_dir, with_time=True, shuffle=False)
    batch = next(iter(DataLoader(ds, batch_size=32)))
    mask = legal_mask(batch)
    expected = float(np.log(float(batch.n_legal.float().mean())))
    B = int(batch.num_graphs)

    # ---------------- shapes and the head's indexing ----------------
    print("\n--- forma e indicizzazione ---")
    torch.manual_seed(0)
    net = ChessMoveGNN(num_event_features=batch.x.shape[1], num_layers=2)
    # eval(): with dropout and batch norm active, two forward passes on the
    # same input give different numbers and the comparison below is meaningless
    net.eval()
    with torch.no_grad():
        logits = net(batch)
    check("output is (B, 4096)", logits.shape == (B, N_CLASSES),
          tuple(logits.shape))
    check("no NaN", not bool(torch.isnan(logits).any()))

    # the head must lay moves out as from*64+to, the same as build_label
    with torch.no_grad():
        h = net.backbone(batch).view(B, 64, -1)
        f, t = net.w_from(h), net.w_to(h)
        grid = torch.bmm(f, t.transpose(1, 2))          # (B, 64, 64)
    ok = all(torch.allclose(logits[b, fr * 64 + to], grid[b, fr, to], atol=1e-5)
             for b in range(min(B, 4)) for fr in (0, 17, 63) for to in (0, 33, 63))
    check("logits[b, from*64+to] is the (from,to) score", ok)

    # ---------------- the attention fix ----------------
    print("\n--- normalise_attention ---")
    torch.manual_seed(0)
    plain = ChessMoveGNN(num_event_features=batch.x.shape[1], num_layers=2,
                         attention_softmax=False)
    _, n_patched = normalise_attention(plain.backbone)
    check("patches every conv of the three paths", n_patched == 3 * 2,
          f"{n_patched} conv")

    # with the softmax the attention weights of each node must sum to 1
    torch.manual_seed(0)
    fixed = ChessMoveGNN(num_event_features=batch.x.shape[1], num_layers=1)
    fixed(batch)
    alpha = fixed.backbone.gat_event[0]._alpha           # (E, heads)
    dst = batch.edge_index[1]
    sums = torch.zeros(batch.x.shape[0]).index_add_(0, dst, alpha[:, 0].detach())
    touched = torch.zeros(batch.x.shape[0], dtype=torch.bool)
    touched[dst] = True
    check("attention sums to 1 on every node that receives edges",
          torch.allclose(sums[touched], torch.ones(int(touched.sum())),
                         atol=1e-4),
          f"min {float(sums[touched].min()):.4f}, "
          f"max {float(sums[touched].max()):.4f}")

    # ---------------- stability, the point of the whole file ----------------
    print("\n--- stabilita' della loss iniziale ---")
    print(f"    attesa tirando a caso fra le mosse legali: {expected:.2f}")

    def worst(layers, fix):
        out = []
        for s in range(args.seeds):
            torch.manual_seed(s)
            n = ChessMoveGNN(num_event_features=batch.x.shape[1],
                             num_layers=layers, attention_softmax=fix)
            lo = n(batch)
            out.append(float(torch.nn.functional.cross_entropy(
                lo + mask, batch.y).detach()))
        return max(out), statistics.median(out)

    for layers in (1, 2, 3):
        w_fix, m_fix = worst(layers, True)
        w_raw, _ = worst(layers, False)
        check(f"layers={layers}: con softmax la loss resta vicino all'attesa",
              w_fix < 3 * expected,
              f"peggiore {w_fix:.2f} (mediana {m_fix:.2f}) "
              f"vs {w_raw:,.1f} senza fix")

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    if all(RESULTS):
        print("il modello e' pronto per il training")
        return 0
    print("something is broken, look at the FAIL lines above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
