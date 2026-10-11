# Run (diagnostic; paths hardcoded, relative to the working directory):
#   python src/chess_eval/diagnostics/diag_ties2.py
#   reads: ../../ckpt/timing/best.pt  and  graphs/test  (first 20 batches)
#
# Checks whether tied moves tie because their squares have IDENTICAL (non-zero)
# embeddings, and whether those squares share the same incoming neighbourhood.
# Hypothesis: the GAT adds no self-loops, so a square's output is built only from
# its neighbours' messages -- its own file/rank never reach it -- and two empty
# squares reached by the same piece through the same edge get the same embedding.

import torch
from torch_geometric.loader import DataLoader
from chess_eval.common.graph_dataset import ParquetGraphDataset, legal_mask
from chess_eval.evaluation.evaluate_models import load_model

model, a, ep, bv = load_model("../../ckpt/timing/best.pt", "cpu")
ds = ParquetGraphDataset("graphs/test", with_time=True,
                         keep_meta=True, shuffle=False)
loader = DataLoader(ds, batch_size=256)

pairs = same_src = same_dst = other = 0
same_h = same_nbh = same_h_and_nbh = 0
esempio = None
with torch.no_grad():
    for i, batch in enumerate(loader):
        if i == 20:
            break
        b = batch.num_graphs
        h = model.backbone(batch).view(b, 64, -1)
        x = batch.x.view(b, 64, -1)
        logits = model(batch) + legal_mask(batch)
        mx = logits.max(dim=1, keepdim=True).values
        at_max = logits == mx
        multi = at_max.sum(dim=1) > 1

        src_all, dst_all = batch.edge_index
        et = batch.edge_type.view(-1)
        per_edge_time = batch.time.numel() == src_all.numel()
        tm = batch.time.view(-1) if per_edge_time else None

        for g in torch.nonzero(multi).view(-1).tolist():
            base = g * 64
            sel = (dst_all >= base) & (dst_all < base + 64)
            es, ed, ety = src_all[sel] - base, dst_all[sel] - base, et[sel]
            etm = tm[sel] if per_edge_time else None

            def vicinato(v):
                m = ed == v
                t = etm[m].tolist() if etm is not None else [0] * int(m.sum())
                return tuple(sorted(zip(es[m].tolist(), ety[m].tolist(), t)))

            idx = torch.nonzero(at_max[g]).view(-1)
            s, d = idx // 64, idx % 64
            for k in range(1, len(idx)):
                pairs += 1
                if s[0] != s[k]:
                    same_dst += int(d[0] == d[k])
                    other += int(d[0] != d[k])
                    continue
                same_src += 1
                u, v = int(d[0]), int(d[k])
                eq_h = torch.equal(h[g, u], h[g, v])
                eq_n = vicinato(u) == vicinato(v)
                same_h += int(eq_h)
                same_nbh += int(eq_n)
                same_h_and_nbh += int(eq_h and eq_n)
                if eq_h and esempio is None:
                    esempio = (int(s[0]), u, v,
                               x[g, u, 9:11].tolist(), x[g, v, 9:11].tolist(),
                               vicinato(u), vicinato(v))

print(f"coppie di mosse a pari merito        : {pairs:,}")
print(f"  stessa casella di partenza         : {same_src:,}  ({same_src/pairs:.1%})")
print(f"  stessa destinazione                : {same_dst:,}")
print(f"  tutto diverso                      : {other:,}")
if same_src:
    print(f"\nfra quelle con la stessa partenza:")
    print(f"  destinazioni con h IDENTICA        : {same_h:,}  ({same_h/same_src:.1%})")
    print(f"  destinazioni con STESSO vicinato   : {same_nbh:,}  ({same_nbh/same_src:.1%})")
    print(f"  entrambe le cose                   : {same_h_and_nbh:,}")
if esempio:
    s0, u, v, xu, xv, nu, nv = esempio
    print(f"\nESEMPIO: dalla casella {s0} verso {u} e {v}")
    print(f"  (colonna, riga) in x : {u} -> {xu}    {v} -> {xv}")
    print(f"  h identica           : si'")
    print(f"  vicinato di {u:<2}       : {nu}")
    print(f"  vicinato di {v:<2}       : {nv}")