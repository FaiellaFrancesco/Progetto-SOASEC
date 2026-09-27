"""
test_model_v2.py

Checks the two model fixes (edge_inputs, self_loops) before any GPU hour is
spent on them, and prints a diagnosis of the environment it runs in.

    python test_model_v2.py -i graphs/test                    (Mac)
    python test_model_v2.py -i /content/graphs/val_timing     (Colab)

THE DIAGNOSIS MATTERS ON ITS OWN. It says whether, in THIS environment and
without the fixes, delta_t and edge types reach the model at all. Run it on
Colab too: that is where v1 was trained, and the answer decides how the v1
"timing" result has to be described in the report.
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from graph_dataset import ParquetGraphDataset, legal_mask
from model import ChessMoveGNN, with_self_loops

RESULTS = []


def check(name, condition, extra=""):
    RESULTS.append(bool(condition))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}"
          f"{'  ' + str(extra) if extra != '' else ''}")


def build(n_features, seed=0, **flags):
    torch.manual_seed(seed)
    net = ChessMoveGNN(num_event_features=n_features, num_layers=3, **flags)
    net.eval()
    return net


"""
INPUT: a model and a batch
OUTPUT: how much the output moves when every delta_t is randomised, and when
        every edge type is randomised. Exactly 0 means the model ignores it.
"""


def sensitivity(net, b):
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        base = net(b)
        bt = b.clone()
        bt.time = torch.rand(bt.time.shape, generator=g) * 20
        be = b.clone()
        be.edge_type = torch.randint(0, 4, be.edge_type.shape, generator=g)
        return (float((net(bt) - base).abs().max()),
                float((net(be) - base).abs().max()))


"""
INPUT: nothing
OUTPUT: a one-graph batch reproducing the case found on the v1 checkpoint: a
        white queen on e8 (square 60), and two empty squares e6 (44) and g6
        (46) that receive one edge each, from e8, of type 'moves', delta_t 20.
        The two differ only in their file column.
"""


def toy_queen():
    x = np.zeros((64, 12), dtype=np.float32)
    for sq in range(64):
        x[sq, 8] = 1.0
        x[sq, 9] = (sq % 8) / 7.0
        x[sq, 10] = (sq // 8) / 7.0
    x[60, 8], x[60, 4], x[60, 6] = 0.0, 1.0, 1.0
    ids = np.zeros(64, dtype=np.int64)
    ids[60] = 1 + 4
    d = Data(x=torch.tensor(x), event_ids=torch.tensor(ids),
             edge_index=torch.tensor([[60, 60], [44, 46]]),
             edge_type=torch.tensor([2, 2]),
             time=torch.tensor([20.0, 20.0]))
    return Batch.from_data_list([d])


def tied(net, b):
    with torch.no_grad():
        logits = net(b) + legal_mask(b)
    mx = logits.max(dim=1, keepdim=True).values
    return int(((logits == mx).sum(dim=1) > 1).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input_dir", required=True,
                   help="cartella parquet costruita con --timing")
    p.add_argument("--batches", type=int, default=8)
    args = p.parse_args()

    ds = ParquetGraphDataset(args.input_dir, with_time=True, shuffle=False)
    batches = [b for _, b in zip(range(args.batches),
                                 DataLoader(ds, batch_size=128))]
    b = batches[0]
    nf = b.x.shape[1]

    # ---------------- 0. diagnosi dell'ambiente ----------------
    print(f"\n--- 0. diagnosi (torch {torch.__version__}, "
          f"pyg {torch_geometric.__version__}) ---")
    dt, de = sensitivity(build(nf), b)
    print(f"  SENZA correzioni: cambiando tutti i delta_t l'output si muove di "
          f"{dt:.6f}")
    print(f"  SENZA correzioni: cambiando tutti i tipi d'arco si muove di "
          f"{de:.6f}")
    if dt == 0 and de == 0:
        print("  => in questo ambiente il modello originale IGNORA tempo degli "
              "archi e tipi d'arco")
    else:
        print("  => in questo ambiente il modello originale li USA gia'")

    # ---------------- 1. edge_inputs ----------------
    print("\n--- 1. edge_inputs: tempo e tipi arrivano all'attenzione ---")
    dt, de = sensitivity(build(nf, edge_inputs=True), b)
    check("con edge_inputs il delta_t cambia l'output", dt > 1e-6, f"{dt:.6f}")
    check("con edge_inputs il tipo d'arco cambia l'output", de > 1e-6,
          f"{de:.6f}")

    # ---------------- 2. self_loops sul caso della donna ----------------
    print("\n--- 2. self_loops: il caso e8 -> e6 / e8 -> g6 ---")
    toy = toy_queen()
    old = build(12, edge_inputs=True)
    new = build(12, edge_inputs=True, self_loops=True)
    with torch.no_grad():
        h_old = old.backbone(toy)
        h_new = new.backbone(with_self_loops(toy, new.self_loop_type))
        l_new = new(toy)[0]
    check("senza loop e6 e g6 hanno h identica (il difetto e' riprodotto)",
          torch.equal(h_old[44], h_old[46]))
    check("con i loop e6 e g6 hanno h diversa",
          not torch.equal(h_new[44], h_new[46]),
          f"diff max {float((h_new[44] - h_new[46]).abs().max()):.4f}")
    check("con i loop le mosse e8-e6 ed e8-g6 hanno punteggi diversi",
          float(l_new[60 * 64 + 44]) != float(l_new[60 * 64 + 46]))

    # ---------------- 3. pareggi su dati veri ----------------
    print(f"\n--- 3. pareggi al massimo su dati veri, pesi casuali ---")
    pos = sum(int(x.num_graphs) for x in batches)
    t0 = sum(tied(build(nf), x) for x in batches)
    t1 = sum(tied(build(nf, edge_inputs=True, self_loops=True), x)
             for x in batches)
    print(f"  {pos:,} posizioni: originale {t0:,} ({t0/pos:.1%}), "
          f"corretto {t1:,} ({t1/pos:.1%})")
    check("con le due correzioni i pareggi spariscono (< 0,5%)",
          t1 < 0.005 * pos)

    # ---------------- 4. struttura dei loop ----------------
    print("\n--- 4. struttura dei self-loop ---")
    net = build(nf, edge_inputs=True, self_loops=True)
    n_e = b.edge_index.shape[1]
    view = with_self_loops(b, net.self_loop_type)
    check("un arco in piu' per casella",
          view.edge_index.shape[1] == n_e + b.x.shape[0],
          f"{n_e:,} -> {view.edge_index.shape[1]:,}")
    check("il batch originale non viene toccato", b.edge_index.shape[1] == n_e)
    check("i loop hanno delta_t 0 e il tipo riservato (4)",
          bool((view.time[n_e:] == 0).all())
          and bool((view.edge_type[n_e:] == 4).all()))
    check("l'embedding dei tipi d'arco ha 5 righe",
          net.backbone.edge_type_emb.num_embeddings == 5)

    # ---------------- 5. addestrabilita' ----------------
    print("\n--- 5. addestrabilita' con le due correzioni ---")
    net.train()
    logits = net(b) + legal_mask(b)
    loss = torch.nn.functional.cross_entropy(logits, b.y.view(-1))
    expected = float(np.log(float(b.n_legal.float().mean())))
    check("perdita iniziale finita e vicina a ln(mosse legali)",
          bool(torch.isfinite(loss)) and float(loss) < 3 * expected,
          f"{float(loss):.3f} (ln mosse legali = {expected:.3f})")
    loss.backward()
    g = net.backbone.edge_type_emb.weight.grad
    used = sorted(set(b.edge_type.tolist())) + [4]
    check("ogni tipo d'arco presente riceve gradiente, self-loop compreso",
          g is not None and all(float(g[k].abs().sum()) > 0 for k in used),
          f"tipi {used}")

    # ---------------- 6. checkpoint ----------------
    print("\n--- 6. checkpoint ---")
    from evaluate_models import load_model
    net.eval()
    common = {"timing": True, "num_layers": 3, "lambda_decay": 0.05,
              "hidden": 64, "heads": 4, "dropout": 0.1,
              "attention_softmax": True}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.pt")
        torch.save({"model": net.state_dict(), "epoch": 1, "best_val": 0.0,
                    "args": dict(common, edge_inputs=True, self_loops=True)},
                   path)
        m = load_model(path, "cpu")[0]
        with torch.no_grad():
            same = torch.allclose(m(b), net(b))
        check("un checkpoint con le correzioni si ricarica identico",
              m.edge_inputs and m.self_loops and same)

        old = build(nf)
        torch.save({"model": old.state_dict(), "epoch": 1, "best_val": 0.0,
                    "args": dict(common)}, path)
        m = load_model(path, "cpu")[0]
        with torch.no_grad():
            same = torch.allclose(m(b), old(b))
        check("un checkpoint vecchio si ricarica come prima (v1 riproducibile)",
              not m.edge_inputs and not m.self_loops and same)

    ok = sum(RESULTS)
    print(f"\n{ok}/{len(RESULTS)} passati")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
