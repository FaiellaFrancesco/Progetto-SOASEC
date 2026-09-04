"""
model.py

The chess model: TimeGNN's DualGATTimeAwareETModel plus the head that turns
per-node embeddings into a move.

    from model import ChessMoveGNN
    net = ChessMoveGNN(num_event_features=13, num_layers=2, lambda_decay=0.05)
    logits = net(batch)                      # (B, 4096)


WHY THERE IS A HEAD AT ALL
--------------------------
TimeGNN predicts the next event, so its output is one vector PER NODE. We have
to predict a move, which is one class out of 4096 PER GRAPH.

The lazy way is to average the 64 squares into one vector and push it through
an MLP. That throws the board away: it asks the network to relearn from scratch
that class 3844 means "from e8 to e1".

But 4096 = 64 x 64. A move IS a pair (from square, to square), and it is the
same convention build_label uses. So the score of a move is computed directly
from the embeddings of the two squares involved:

    logits[b, f*64 + t] = <Wf . h[b,f] , Wt . h[b,t]>

33x fewer parameters than pooling+MLP, and the model starts out knowing that a
move is a relation between two squares, which is exactly what the graph encodes.


THE ATTENTION FIX
-----------------
TimeAwareETGATConv.message() never normalises the attention: in the library's
source the softmax line is commented out. The guide, however, describes the
decay as being applied "prima della normalizzazione softmax" - so the softmax
is part of the intended design and its absence looks like an oversight.

Without it, each layer multiplies the scale of the previous one. Measured over
8 seeds on real data, the initial loss (which should sit near ln(legal moves)
~ 3.3):

    layers=1        2.6 - 3.6          fine
    layers=2        2.6 - 416          unusable
    layers=3        3.0 - 1.8e14       catastrophic

normalise_attention() restores that one line on an already-built model. It does
not modify the library: it rebinds the method on the instances, so the import
stays exactly the one the project brief asks for.

Keep it on. attention_softmax=False is there only to reproduce the library's
own behaviour for the report.
"""

import types

import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax

from timegnn.models.gat_time_decay_status_emb import DualGATTimeAwareETModel

N_SQUARES = 64
N_CLASSES = 4096


"""
INPUT: self and the arguments PyG passes to message()
OUTPUT: the message, with the attention normalised over each node's neighbours

Same body as the library's message(), plus the softmax line. Normalising AFTER
the decay is what the guide prescribes: the decay reweights the logits, the
softmax turns them into shares that sum to 1 per node.
"""


def _message_with_softmax(self, x_i=None, x_j=None, time=None, edge_attr=None,
                          edge_index=None, index=None, ptr=None, size_i=None,
                          alpha=None, **kwargs):
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
        # THE MISSING LINE: shares per node instead of unbounded weights
        alpha = pyg_softmax(alpha, index, ptr, num_nodes=size_i)

    self._alpha = alpha
    return x_j * alpha.unsqueeze(-1)


"""
INPUT: a DualGATTimeAwareETModel
OUTPUT: the same object, with every conv normalising its attention

Rebinds message() on each TimeAwareETGATConv instance. The library's files are
untouched, so `import timegnn` still loads the professor's code as-is.
"""


def normalise_attention(model):
    n = 0
    for path in (model.gat_embed, model.gat_event, model.gat_concat):
        for conv in path:
            conv.message = types.MethodType(_message_with_softmax, conv)
            n += 1
    return model, n


class ChessMoveGNN(nn.Module):
    """
    DualGATTimeAwareETModel + bilinear move head.

    The GAT hyperparameters keep the library's own names so the two can be
    compared line by line. The defaults follow the project brief where they do
    not clash with what we measured: hidden dims 128-256 come from the brief,
    num_layers=2 is a compromise between the brief (3-5) and the library's
    guidance (1-2), which our stability measurements back.
    """

    def __init__(self, num_event_features, num_piece_ids=13, num_edge_types=4,
                 embedding_dims=64, gat_hidden_dim_event=64,
                 gat_hidden_dim_embed=64, gat_hidden_dim_concat=64,
                 node_out_dim=64, num_heads=4, edge_type_dim=16,
                 lambda_decay=0.05, num_layers=2, dropout=0.1,
                 use_batch_norm=True, attention_softmax=True):
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
            num_edge_types=num_edge_types,
            edge_type_dim=edge_type_dim,
            lambda_decay=lambda_decay,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        self.attention_softmax = attention_softmax
        if attention_softmax:
            normalise_attention(self.backbone)

        # the bilinear head: one projection for the source square, one for the
        # destination, and their dot product is the score of that move
        self.w_from = nn.Linear(node_out_dim, node_out_dim, bias=False)
        self.w_to = nn.Linear(node_out_dim, node_out_dim, bias=False)

    def forward(self, batch):
        h = self.backbone(batch)                       # (B*64, node_out_dim)
        b = h.shape[0] // N_SQUARES
        h = h.view(b, N_SQUARES, -1)
        f = self.w_from(h)                             # (B, 64, D)
        t = self.w_to(h)
        # (B, 64, 64) -> (B, 4096), index f*64+t: the same order as build_label
        return torch.bmm(f, t.transpose(1, 2)).reshape(b, N_CLASSES)
