# Run (diagnostic; paths hardcoded, relative to the working directory):
#   python src/chess_eval/diagnostics/diag_ties.py
#   reads: ../../ckpt/timing/best.pt  and  graphs/test  (first 20 batches)
#
# Checks whether the exact logit ties come from node embeddings that are all
# zero. The move head has no bias, so logit(src -> dst) = <W_from h[src],
# W_to h[dst]>: if a square's embedding h is the zero vector, every move to or
# from it scores exactly 0.0. Two tells: the tied maximum is exactly 0.0, and
# the tied moves land on squares whose embedding is zero.

import torch
from torch_geometric.loader import DataLoader
from chess_eval.common.graph_dataset import ParquetGraphDataset, legal_mask
from chess_eval.evaluation.evaluate_models import load_model

model, a, ep, bv = load_model("../../ckpt/timing/best.pt", "cpu")
ds = ParquetGraphDataset("graphs/test", with_time=True,
                         keep_meta=True, shuffle=False)
loader = DataLoader(ds, batch_size=256)

nodes = dead = empty = dead_empty = occ = dead_occ = 0
pos = tied = tied_zero = tied_dead_dst = tied_dead_src = 0
with torch.no_grad():
    for i, batch in enumerate(loader):
        if i == 20:
            break
        b = batch.num_graphs
        h = model.backbone(batch).view(b, 64, -1)
        is_dead = (h == 0).all(dim=-1)                      # (b, 64)
        is_empty = (batch.x[:, 8] > 0.5).view(b, 64)
        nodes += is_dead.numel()
        dead += int(is_dead.sum())
        empty += int(is_empty.sum())
        dead_empty += int((is_dead & is_empty).sum())
        occ += int((~is_empty).sum())
        dead_occ += int((is_dead & ~is_empty).sum())

        logits = model(batch) + legal_mask(batch)
        mx = logits.max(dim=1, keepdim=True).values
        at_max = logits == mx                               # (b, 4096)
        multi = at_max.sum(dim=1) > 1
        pos += b
        tied += int(multi.sum())
        tied_zero += int((multi & (mx.view(-1) == 0)).sum())
        for g in torch.nonzero(multi).view(-1).tolist():
            idx = torch.nonzero(at_max[g]).view(-1)
            src, dst = idx // 64, idx % 64
            tied_dead_dst += int(bool(is_dead[g, dst].all()))
            tied_dead_src += int(bool(is_dead[g, src].all()))

print(f"posizioni esaminate: {pos:,}\n")
print("EMBEDDING DELLE CASELLE")
print(f"  caselle con h tutto zero      : {dead:,} / {nodes:,}  ({dead/nodes:.1%})")
print(f"    fra le caselle vuote        : {dead_empty/empty:.1%}")
print(f"    fra le caselle occupate     : {dead_occ/max(occ,1):.1%}")
print("\nPAREGGI AL MASSIMO")
print(f"  posizioni con pareggio        : {tied:,}  ({tied/pos:.1%})   [atteso ~34%]")
if tied:
    print(f"  ...col massimo esattamente 0.0: {tied_zero:,}  ({tied_zero/tied:.1%})")
    print(f"  ...destinazioni tutte morte   : {tied_dead_dst:,}  ({tied_dead_dst/tied:.1%})")
    print(f"  ...partenze tutte morte       : {tied_dead_src:,}  ({tied_dead_src/tied:.1%})")