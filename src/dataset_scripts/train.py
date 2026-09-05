"""
train.py


    python train.py --train graphs/train_timing --val graphs/val_timing \\
                    --timing --out /content/drive/MyDrive/SOASEC/ckpt/timing

    python train.py --train graphs/train_plain  --val graphs/val_plain \\
                    --out /content/drive/MyDrive/SOASEC/ckpt/plain

Every epoch writes a checkpoint, and re-running the exact same command resumes
from it: same epoch counter, same optimizer state, same best score, same
history. A dropped session costs one epoch, not the whole run. Put --out on
Drive, because /content does not survive a disconnection.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from graph_dataset import ParquetGraphDataset, legal_mask
from model import ChessMoveGNN

MAX_DEPTH = 8          # buckets reported for n_remaining


"""
INPUT: the model, a DataLoader, the device, and optionally an optimizer
OUTPUT: (mean loss, overall accuracy, accuracy per n_remaining, n examples)

One pass over the data. With an optimizer it trains, without it evaluates.
"""


def run_epoch(model, loader, device, optimizer=None, log_every=200):
    train = optimizer is not None
    model.train(train)

    tot_loss = tot_n = tot_correct = 0
    per_depth = {d: [0, 0] for d in range(1, MAX_DEPTH + 1)}   # [correct, seen]
    t0 = time.time()

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        mask = legal_mask(batch).to(device)

        with torch.set_grad_enabled(train):
            logits = model(batch)
            loss = torch.nn.functional.cross_entropy(logits + mask, batch.y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            # the attention fix keeps activations sane, but a single odd batch
            # should not be able to blow up the weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        n = int(batch.num_graphs)
        pred = (logits + mask).argmax(dim=1)
        correct = (pred == batch.y)

        tot_loss += float(loss.detach()) * n
        tot_n += n
        tot_correct += int(correct.sum())

        for d, c in zip(batch.n_remaining.tolist(), correct.tolist()):
            if d in per_depth:
                per_depth[d][0] += int(c)
                per_depth[d][1] += 1

        if train and log_every and i % log_every == 0 and i:
            print(f"    batch {i:>5}  loss {tot_loss/tot_n:.4f}  "
                  f"acc {tot_correct/tot_n:.3%}  "
                  f"{tot_n/(time.time()-t0):,.0f} ex/s", flush=True)

    acc_depth = {d: (c / s if s else None) for d, (c, s) in per_depth.items()}
    return tot_loss / max(tot_n, 1), tot_correct / max(tot_n, 1), acc_depth, tot_n


"""
INPUT: the checkpoint directory
OUTPUT: the loaded state dict, or None

Resume is by directory, not by flag: re-running the same command picks up where
it left off
"""


def load_checkpoint(out_dir, model, optimizer, device):
    path = os.path.join(out_dir, "last.pt")
    if not os.path.isfile(path):
        return None
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    print(f"ripreso da {path}: epoca {ck['epoch']}, "
          f"miglior val loss {ck['best_val']:.4f}")
    return ck


def save_checkpoint(out_dir, name, model, optimizer, epoch, best_val, history,
                    args):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "history": history,
                # the hyperparameters travel with the weights: a checkpoint
                # you cannot rebuild the model from is useless
                "args": vars(args)},
               os.path.join(out_dir, name))


def fmt_depth(acc):
    parts = [f"n{d}={a:.1%}" for d, a in sorted(acc.items())
             if a is not None]
    return "  ".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="cartella parquet di training")
    p.add_argument("--val", required=True, help="cartella parquet di validation")
    p.add_argument("--out", required=True,
                   help="dove salvare i checkpoint (mettila su Drive)")
    p.add_argument("--timing", action="store_true",
                   help="usa delta_t vero; senza, time=0 e il decadimento e' inerte")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--lambda-decay", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=7,
                   help="epoche senza miglioramento prima di fermarsi")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-attention-softmax", dest="attention_softmax",
                   action="store_false",
                   help="riproduce il comportamento originale della libreria")
    p.set_defaults(attention_softmax=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = ParquetGraphDataset(args.train, with_time=args.timing,
                                   shuffle=True, seed=args.seed)
    val_ds = ParquetGraphDataset(args.val, with_time=args.timing,
                                 shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.workers)

    n_features = 13 if args.timing else 12
    model = ChessMoveGNN(num_event_features=n_features,
                         num_layers=args.num_layers,
                         lambda_decay=args.lambda_decay,
                         gat_hidden_dim_event=args.hidden,
                         gat_hidden_dim_embed=args.hidden,
                         gat_hidden_dim_concat=args.hidden,
                         num_heads=args.heads,
                         dropout=args.dropout,
                         attention_softmax=args.attention_softmax).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"device      : {device}")
    print(f"timing      : {args.timing}  (x ha {n_features} colonne)")
    print(f"layers      : {args.num_layers}   softmax attenzione: "
          f"{args.attention_softmax}")
    print(f"parametri   : {sum(q.numel() for q in model.parameters()):,}")
    print(f"checkpoint  : {args.out}")

    ck = load_checkpoint(args.out, model, optimizer, device)
    start_epoch = ck["epoch"] + 1 if ck else 1
    best_val = ck["best_val"] if ck else float("inf")
    history = ck["history"] if ck else []
    bad_epochs = 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)      # senza, ogni epoca vede lo stesso ordine
        t0 = time.time()
        print(f"\n--- epoca {epoch}/{args.epochs} ---", flush=True)

        tr_loss, tr_acc, _, n_tr = run_epoch(model, train_loader, device,
                                             optimizer)
        va_loss, va_acc, va_depth, n_va = run_epoch(model, val_loader, device)

        dt = time.time() - t0
        print(f"  train  loss {tr_loss:.4f}  acc {tr_acc:.2%}  ({n_tr:,} es.)")
        print(f"  val    loss {va_loss:.4f}  acc {va_acc:.2%}  ({n_va:,} es.)")
        print(f"  val per profondita': {fmt_depth(va_depth)}")
        print(f"  {dt/60:.1f} min")

        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "train_acc": tr_acc, "val_loss": va_loss,
                        "val_acc": va_acc,
                        "val_acc_by_depth": {str(k): v for k, v
                                             in va_depth.items()},
                        "seconds": dt})

        # last.pt PRIMA di ogni altra cosa: e' quello che permette di riprendere
        save_checkpoint(args.out, "last.pt", model, optimizer, epoch,
                        min(best_val, va_loss), history, args)
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

        if va_loss < best_val:
            best_val = va_loss
            bad_epochs = 0
            save_checkpoint(args.out, "best.pt", model, optimizer, epoch,
                            best_val, history, args)
            print(f"  nuovo best (val loss {best_val:.4f}) salvato")
        else:
            bad_epochs += 1
            print(f"  nessun miglioramento da {bad_epochs} epoche "
                  f"(best {best_val:.4f})")
            if bad_epochs >= args.patience:
                print(f"\nearly stopping: {args.patience} epoche senza "
                      f"miglioramento")
                break

    print(f"\nfinito. miglior val loss {best_val:.4f}")
    print(f"pesi migliori: {os.path.join(args.out, 'best.pt')}")


if __name__ == "__main__":
    main()
