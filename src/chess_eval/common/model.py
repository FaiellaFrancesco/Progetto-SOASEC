# Library module (imported, not run directly).
# Used by: training.train, evaluation.evaluate_models, tests.test_model, tests.test_model_v2
#
# The chess model: TimeGNN's DualGATTimeAwareETModel plus a head that turns
# per-node embeddings into a move.
#
#   from chess_eval.common.model import ChessMoveGNN
#   net = ChessMoveGNN(num_event_features=13, num_layers=2, lambda_decay=0.05)
#   logits = net(batch)                      # (B, 4096)
#
# The head: TimeGNN outputs one vector per node, but we need one move (a class
# out of 4096) per graph. Since 4096 = 64*64 and a move is a (from, to) pair,
# the score of a move is a bilinear form on the two squares' embeddings
# (logits[b, f*64+t] = <Wf.h[b,f], Wt.h[b,t]>), the same convention build_label
# uses. ~33x fewer params than pooling+MLP, and it starts out knowing a move is
# a relation between two squares.
#
# The attention fix: TimeAwareETGATConv.message() never normalises the
# attention (the softmax line is commented out in the library), so each layer
# multiplies the previous scale and 2+ layers diverge. normalise_attention()
# rebinds message() on the instances to restore the softmax, without touching
# the library's files. Keep it on; attention_softmax=False only reproduces the
# library's own behaviour for the report.

import types

import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax

from timegnn.models.gat_time_decay_status_emb import DualGATTimeAwareETModel

N_SQUARES = 64
N_CLASSES = 4096


def _message_with_softmax(self, x_i=None, x_j=None, time=None, edge_attr=None,
                          edge_index=None, index=None, ptr=None, size_i=None,
                          alpha=None, **kwargs):
    """The library's message(), plus the softmax that normalises attention.

    Normalising after the decay is what the guide prescribes: the decay
    reweights the logits, the softmax turns them into per-node shares.
    """
    if x_j is None:
        x_j = kwargs.get("x_j")
    if x_j is None:
        return None
    if x_i is None:
        x_i = kwargs.get("x_i")
    if time is None:
        time = kwargs.get("time")
    if edge_attr is None:
        edge_attr = kwargs.get("edge_attr")
    if edge_index is None:
        edge_index = kwargs.get("edge_index")

    if x_i is None:
        alpha = torch.ones(x_j.size(0), x_j.size(1), device=x_j.device)
    else:
        if time is None:
            time = 0.0
        alpha = self.edge_attention(x_i, x_j, time, edge_attr, edge_index)
        # the missing line: shares per node instead of unbounded weights
        alpha = pyg_softmax(alpha, index, ptr, num_nodes=size_i)

    self._alpha = alpha
    return x_j * alpha.unsqueeze(-1)


def normalise_attention(model):
    """Rebind message() on every conv so it normalises its attention.

    The library's files stay untouched, so `import timegnn` still loads the
    professor's code as-is.
    """
    n = 0
    for path in (model.gat_embed, model.gat_event, model.gat_concat):
        for conv in path:
            conv.message = types.MethodType(_message_with_softmax, conv)
            n += 1
    return model, n


class ChessMoveGNN(nn.Module):
    """DualGATTimeAwareETModel + bilinear move head.

    The GAT hyperparameters keep the library's own names so the two can be
    compared line by line. Defaults follow the project brief where they do not
    clash with what we measured: hidden dims 128-256 from the brief, and
    num_layers=2 as a compromise between the brief (3-5) and the library (1-2),
    which our stability measurements back.
    """

    def __init__(self, num_event_features, num_piece_ids=13, num_edge_types=4,
                 embedding_dims=64, gat_hidden_dim_event=64,
                 gat_hidden_dim_embed=64, gat_hidden_dim_concat=64,
                 node_out_dim=64, num_heads=4, edge_type_dim=16,
                 lambda_decay=0.05, num_layers=2, dropout=0.1,
                 use_batch_norm=True, attention_softmax=True,
                 edge_inputs=False, self_loops=False):
        super().__init__()
        self.backbone = DualGATTimeAwareETModel(
            num_event_features=num_event_features,
            num_embedding_features=num_piece_ids,
            embedding_dims=embedding_dims,
            gat_hidden_dim_event=gat_hidden_dim_event,
            gat_hidden_dim_embed=gat_hidden_dim_embed,
            gat_hidden_dim_concat=gat_hidden_dim_concat,
            output_dim=node_out_dim,
            num_heads=num_heads,
            # one extra type, reserved for the self-loops
            num_edge_types=num_edge_types + (1 if self_loops else 0),
            edge_type_dim=edge_type_dim,
            lambda_decay=lambda_decay,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        self.attention_softmax = attention_softmax
        if attention_softmax:
            normalise_attention(self.backbone)

        # the two fixes: see restore_edge_inputs() and with_self_loops()
        self.edge_inputs = edge_inputs
        if edge_inputs:
            restore_edge_inputs(self.backbone)
        self.self_loops = self_loops
        self.self_loop_type = num_edge_types

        # bilinear head: project the source and destination squares, their dot
        # product is the move's score
        self.w_from = nn.Linear(node_out_dim, node_out_dim, bias=False)
        self.w_to = nn.Linear(node_out_dim, node_out_dim, bias=False)

    def forward(self, batch):
        if self.self_loops:
            batch = with_self_loops(batch, self.self_loop_type)
        h = self.backbone(batch)                       # (B*64, node_out_dim)
        b = h.shape[0] // N_SQUARES
        h = h.view(b, N_SQUARES, -1)
        f = self.w_from(h)                             # (B, 64, D)
        t = self.w_to(h)
        # (B, 64, 64) -> (B, 4096), index f*64+t: same order as build_label
        return torch.bmm(f, t.transpose(1, 2)).reshape(b, N_CLASSES)


# --------------------------------------------------------------------------
# The two fixes (off by default: see ChessMoveGNN's edge_inputs / self_loops)
# --------------------------------------------------------------------------

from types import SimpleNamespace


def _forward_with_edge_inputs(self, x, edge_index, edge_attr=None, time=None,
                              return_attention=False):
    """The library's forward(), but actually handing time and edge_attr to propagate().

    Since PyG 2.5 the library's forward drops both for this class (its template
    generation fails and PyG falls back to propagate(edge_index, size,
    **kwargs)), so the attention never saw the edge types or the time decay.
    This body is the library's, minus that check.
    """
    h, c = self.heads, self.out_channels
    x = self.lin(x).view(-1, h, c)
    out = self.propagate(edge_index, x=x, edge_attr=edge_attr, time=time,
                         alpha=None, size=None)
    out = out.view(-1, h * c) if self.concat else out.mean(dim=1)
    if return_attention:
        return out, self._alpha
    return out


def restore_edge_inputs(model):
    """Rebind forward() on each conv so it really receives time and edge types.

    Same idea as normalise_attention: the library's files stay untouched.
    """
    n = 0
    for path in (model.gat_embed, model.gat_event, model.gat_concat):
        for conv in path:
            conv.forward = types.MethodType(_forward_with_edge_inputs, conv)
            n += 1
    return model, n


def with_self_loops(batch, loop_type):
    """A view of the batch with one extra edge i->i per node (type loop_type, delta_t 0).

    The library never calls add_self_loops, so without these a square's own
    features never reach its own embedding: two empty squares reached by the
    same piece get the same embedding and every move to them ties. Added at
    forward time, so the parquet and the library stay untouched and the original
    batch is not modified (legal_mask and the labels still read it as it was).
    delta_t=0 gives decay exp(0)=1, so the timed and untimed models still differ
    only in time.
    """
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