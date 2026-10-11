# Run (one-off: applies the two model fixes; ALREADY applied in this repo):
#   python src/chess_eval/tools/patch_model_v2.py
#
# Two fixes, each behind its own flag, across three files:
#   model.py            ChessMoveGNN(..., edge_inputs=False, self_loops=False)
#   train.py            new flags --edge-inputs and --self-loops
#   evaluate_models.py  load_model() reads both from the checkpoint's args
# It saves a .bak of every file it touches, refuses a file it does not
# recognise, and is a no-op on a second run.
#
# fix 1 (edge_inputs): delta_t and edge types never reached the attention.
#   TimeAwareETGATConv.forward only forwards time/edge_attr if propagate's
#   signature names them; since PyG 2.5 the template generation fails for this
#   class, PyG falls back to propagate(edge_index, size, **kwargs), and both are
#   silently dropped (changing every delta_t or edge type moved the output by 0).
# fix 2 (self_loops): a square never saw its own features.
#   The same forward never calls add_self_loops, so a square's own piece/colour/
#   file/rank never reach its embedding (on v1, 97% of same-source tied moves had
#   bit-identical destination embeddings even with a different file).
# Without the flags everything behaves as before and old checkpoints load as
# before, so the v1 results stay reproducible.
#
# NOTE: written for the old flat layout (model.py/train.py/evaluate_models.py in
# one folder). After the package split it no longer applies as-is; kept only as
# a record of what the two fixes changed.

import os
import shutil
import sys

HELPERS = '''

# --------------------------------------------------------------------------
# The two fixes (off by default: see ChessMoveGNN's edge_inputs / self_loops)
# --------------------------------------------------------------------------

from types import SimpleNamespace


"""
INPUT: self (a TimeAwareETGATConv) and the arguments of its forward
OUTPUT: the same output as the library's forward, but with time and edge_attr
        actually handed to propagate()

The library's forward passes them only if propagate's signature names them.
Since PyG 2.5 it does not for this class (template generation fails and PyG
falls back to propagate(edge_index, size, **kwargs)), so both were silently
dropped and the attention never saw the edge types or the time decay. This body
is the library's, minus that check.
"""


def _forward_with_edge_inputs(self, x, edge_index, edge_attr=None, time=None,
                              return_attention=False):
    h, c = self.heads, self.out_channels
    x = self.lin(x).view(-1, h, c)
    out = self.propagate(edge_index, x=x, edge_attr=edge_attr, time=time,
                         alpha=None, size=None)
    out = out.view(-1, h * c) if self.concat else out.mean(dim=1)
    if return_attention:
        return out, self._alpha
    return out


"""
INPUT: a DualGATTimeAwareETModel
OUTPUT: the same object, with every conv really receiving time and edge types

Rebinds forward() on each conv instance, like normalise_attention does for
message(): the library's files stay untouched.
"""


def restore_edge_inputs(model):
    n = 0
    for path in (model.gat_embed, model.gat_event, model.gat_concat):
        for conv in path:
            conv.forward = types.MethodType(_forward_with_edge_inputs, conv)
            n += 1
    return model, n


"""
INPUT: a PyG batch and the edge-type index reserved for self-loops
OUTPUT: a light view of the batch with one extra edge i -> i on every node,
        of type loop_type and delta_t 0

The library's forward never calls add_self_loops, so without these edges a
square's own features never reach its own embedding: two empty squares reached
by the same single piece receive the same message, get the same embedding, and
every move to them ties. The loops are added at forward time, so the parquet
files and the library stay untouched, and the original batch is not modified
(legal_mask and the labels still read it as it was).

delta_t = 0 gives the loop the decay exp(0) = 1. In the model without timing
every edge already has delta_t = 0, so the two models still differ only in time.
"""


def with_self_loops(batch, loop_type):
    n = batch.x.size(0)
    dev = batch.x.device
    loop = torch.arange(n, device=dev)
    edge_index = torch.cat([batch.edge_index, torch.stack([loop, loop])], dim=1)
    edge_type = torch.cat([
        batch.edge_type.view(-1),
        torch.full((n,), loop_type, dtype=batch.edge_type.dtype, device=dev)])
    time = torch.cat([
        batch.time.view(-1),
        torch.zeros(n, dtype=batch.time.dtype, device=dev)])
    return SimpleNamespace(x=batch.x, event_ids=batch.event_ids,
                           edge_index=edge_index, edge_type=edge_type,
                           time=time, batch=getattr(batch, "batch", None))
'''

MODEL_EDITS = [
    ("                 use_batch_norm=True, attention_softmax=True):\n",
     "                 use_batch_norm=True, attention_softmax=True,\n"
     "                 edge_inputs=False, self_loops=False):\n"),
    ("            num_edge_types=num_edge_types,\n",
     "            # one extra type, reserved for the self-loops\n"
     "            num_edge_types=num_edge_types + (1 if self_loops else 0),\n"),
    ("            normalise_attention(self.backbone)\n",
     "            normalise_attention(self.backbone)\n"
     "\n"
     "        # the two fixes: see restore_edge_inputs() and with_self_loops()\n"
     "        self.edge_inputs = edge_inputs\n"
     "        if edge_inputs:\n"
     "            restore_edge_inputs(self.backbone)\n"
     "        self.self_loops = self_loops\n"
     "        self.self_loop_type = num_edge_types\n"),
    ("        h = self.backbone(batch)                       # (B*64, node_out_dim)\n",
     "        if self.self_loops:\n"
     "            batch = with_self_loops(batch, self.self_loop_type)\n"
     "        h = self.backbone(batch)                       # (B*64, node_out_dim)\n"),
]


def patch(path, edits, append=None, marker=None):
    """Apply (old, new) exact replacements to a file, saving a .bak alongside.

    Every 'old' must occur exactly once or nothing is written: a file that does
    not look like the one this patch targets is left alone. `marker` means the
    file is already patched; `append` is optional text added at the end.
    """
    src = open(path).read()
    if marker in src:
        print(f"{path}: gia' modificato, salto")
        return
    for old, _ in edits:
        n = src.count(old)
        if n != 1:
            sys.exit(f"{path}: mi aspettavo 1 occorrenza di\n  {old!r}\n"
                     f"trovate {n}. Non tocco niente.")
    out = src
    for old, new in edits:
        out = out.replace(old, new)
    if append:
        out = out.rstrip("\n") + "\n" + append
    shutil.copy(path, path + ".bak")
    open(path, "w").write(out)
    print(f"{path}: modificato (originale in {path}.bak)")


def line_with(path, text):
    """Return the whole line of `path` containing `text`, plus its indentation."""
    line = next((l for l in open(path).read().splitlines(True) if text in l),
                None)
    if line is None:
        sys.exit(f"{path}: non trovo {text!r}. Non tocco niente.")
    return line, line[:len(line) - len(line.lstrip())]


def main():
    for f in ("model.py", "train.py", "evaluate_models.py"):
        if not os.path.isfile(f):
            sys.exit(f"non trovo {f}: lancialo da src/dataset_scripts")

    patch("model.py", MODEL_EDITS, append=HELPERS, marker="restore_edge_inputs")

    # train.py: two flags, and both passed to the model
    if "--edge-inputs" in open("train.py").read():
        print("train.py: gia' modificato, salto")
    else:
        patch_train()
    if "edge_inputs" in open("evaluate_models.py").read():
        print("evaluate_models.py: gia' modificato, salto")
    else:
        patch_evaluate()

    print("\nfatto. Ora: python test_model_v2.py -i graphs/test")


def patch_train():
    old_call, ind = line_with(
        "train.py", "attention_softmax=args.attention_softmax).to(device)")
    new_call = (f"{ind}attention_softmax=args.attention_softmax,\n"
                f"{ind}edge_inputs=args.edge_inputs,\n"
                f"{ind}self_loops=args.self_loops).to(device)\n")
    patch("train.py", [
        ("    p = argparse.ArgumentParser()\n",
         "    p = argparse.ArgumentParser()\n"
         "    p.add_argument(\"--edge-inputs\", action=\"store_true\",\n"
         "                   help=\"passa davvero delta_t e tipi di arco \"\n"
         "                        \"all'attenzione: model.restore_edge_inputs\")\n"
         "    p.add_argument(\"--self-loops\", action=\"store_true\",\n"
         "                   help=\"un self-loop su ogni casella: \"\n"
         "                        \"model.with_self_loops\")\n"),
        (old_call, new_call),
    ], marker="--edge-inputs")


def patch_evaluate():
    # the flags come from the checkpoint; old checkpoints have neither key and
    # load exactly as before
    old_arg, ind = line_with(
        "evaluate_models.py", 'attention_softmax=a.get("attention_softmax", True),')
    patch("evaluate_models.py", [
        (old_arg, old_arg
         + f'{ind}edge_inputs=a.get("edge_inputs", False),\n'
         + f'{ind}self_loops=a.get("self_loops", False),\n'),
    ], marker="edge_inputs")


if __name__ == "__main__":
    main()