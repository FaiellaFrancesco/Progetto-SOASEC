# Run (diagnostic; paths hardcoded, relative to the working directory):
#   python src/chess_eval/diagnostics/diag_topk.py
#   reads: ../../ckpt/timing/best.pt  and  graphs/test  (first 20 batches)
#
# Checks why argmax and topk[0] disagree. If the two moves have the SAME logit,
# it is a tie the two functions just break differently; if topk[0] has a LOWER
# logit, then topk is not returning its columns in descending order and column 0
# is not the maximum.

import torch
from torch_geometric.loader import DataLoader
from chess_eval.common.graph_dataset import ParquetGraphDataset, legal_mask
from chess_eval.evaluation.evaluate_models import load_model

print("torch", torch.__version__)
model, a, ep, bv = load_model("../../ckpt/timing/best.pt", "cpu")
ds = ParquetGraphDataset("graphs/test", with_time=True,
                         keep_meta=True, shuffle=False)
loader = DataLoader(ds, batch_size=256)

tot = diff = tie = lower = higher = multimax = unsorted = 0
with torch.no_grad():
    for i, batch in enumerate(loader):
        if i == 20:
            break
        logits = model(batch) + legal_mask(batch)
        arg = logits.argmax(dim=1)
        tk = logits.topk(5, dim=1)
        top = tk.indices[:, 0]
        va = logits.gather(1, arg.view(-1, 1)).view(-1)
        vt = logits.gather(1, top.view(-1, 1)).view(-1)
        d = arg != top
        tot += logits.size(0)
        diff += int(d.sum())
        tie += int((d & (va == vt)).sum())
        lower += int((d & (vt < va)).sum())
        higher += int((d & (vt > va)).sum())
        mx = logits.max(dim=1, keepdim=True).values
        multimax += int(((logits == mx).sum(dim=1) > 1).sum())
        unsorted += int((tk.values[:, :-1] < tk.values[:, 1:]).any(dim=1).sum())

print(f"esempi esaminati                        : {tot:,}")
print(f"argmax != topk[0]                       : {diff:,}  ({diff/tot:.1%})")
print(f"  ...con logit IDENTICO  (ipotesi A)    : {tie:,}")
print(f"  ...con topk[0] PIU' BASSO (ipotesi B) : {lower:,}")
print(f"  ...con topk[0] piu' alto (assurdo)    : {higher:,}")
print(f"righe con piu' di un massimo            : {multimax:,}  ({multimax/tot:.1%})")
print(f"righe in cui topk non e' decrescente    : {unsorted:,}")