# Run (quick diagnostic; paths are hardcoded, relative to the working directory):
#   python src/chess_eval/evaluation/check_top1.py
#   reads: ../../ckpt/timing/best.pt  and  graphs/test
#
# Sanity check on the timing model: computes overall top-1 accuracy two ways in
# one pass -- argmax vs topk(5)[:, 0] -- and counts how many examples the two
# disagree on. If they differ, tie-handling is the cause; if they agree but
# neither is ~26.62%, something else moved.

import torch
from torch_geometric.loader import DataLoader
from chess_eval.common.graph_dataset import ParquetGraphDataset, legal_mask
from chess_eval.evaluation.evaluate_models import load_model

device = "cpu"
model, a, ep, bv = load_model("../../ckpt/timing/best.pt", device)
print(f"checkpoint: epoca {ep}, val loss {bv:.4f}, timing {a['timing']}, "
      f"layers {a['num_layers']}, softmax {a.get('attention_softmax', True)}")

ds = ParquetGraphDataset("graphs/test", with_time=True,
                         keep_meta=True, shuffle=False)
loader = DataLoader(ds, batch_size=256)

n = ok_arg = ok_top = disagree = 0
with torch.no_grad():
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        logits = model(batch) + legal_mask(batch).to(device)
        arg = logits.argmax(dim=1)
        top = logits.topk(5, dim=1).indices[:, 0]
        y = batch.y.view(-1)
        n += y.numel()
        ok_arg += int((arg == y).sum())
        ok_top += int((top == y).sum())
        disagree += int((arg != top).sum())
        if i % 100 == 0:
            print(f"{n:>7,}  argmax {ok_arg/n:7.4%}  topk[0] {ok_top/n:7.4%}"
                  f"  discordi {disagree}", flush=True)

print(f"\nFINALE su {n:,} puzzle")
print(f"  argmax  : {ok_arg/n:.4%}  ({ok_arg:,})")
print(f"  topk[0] : {ok_top/n:.4%}  ({ok_top:,})")
print(f"  esempi su cui i due metodi non concordano: {disagree:,}")
print(f"  atteso dal 9 settembre: 26.6216%  (51,502)")