"""
graph_dataset.py

The bridge between db_rows_to_graphs.py (parquet) and TimeGNN. Nothing else in
the project converts one into the other.

    from graph_dataset import ParquetGraphDataset
    from torch_geometric.loader import DataLoader

    ds = ParquetGraphDataset("./graphs/train_timing", with_time=True)
    for batch in DataLoader(ds, batch_size=64):
        out = model(batch)          # DualGATTimeAwareETModel takes the batch


THE CONTRACT, READ OFF THE LIBRARY'S OWN SOURCE
------------------------------------------------
DualGATTimeAwareETModel.forward(data_event) uses exactly these fields:

    edge_type = data_event.edge_type          # [E]  int, index into an nn.Embedding
    edge_attr = self.edge_type_emb(edge_type) # the model builds edge_attr ITSELF
    time      = data_event.time               # [E]  float, the delta_t
    edge_index= data_event.edge_index         # [2, E]
    x_embed   = self.embedding(data_event.event_ids.view(-1))   # [N] int
    x_event   = data_event.x                  # [N, num_event_features]

Three consequences, and they are the whole reason this file was rewritten:

1. time is a SEPARATE field, not a column of edge_attr. The published guide
   shows `time_diff = edge_attr[:, 0]`, but that is the older standalone
   script; the packaged library takes `time=` as its own argument.

2. edge_type is an INTEGER INDEX per edge, not a one-hot row. It is fed to
   nn.Embedding, which needs indices. Our encoder writes a 4-column one-hot,
   so argmax converts it back.

3. We do NOT pass edge_attr at all. The model embeds the edge type on its own.

event_ids is the piece standing on each square, 0..12 (0 = empty, 1-6 white
pawn..king, 7-12 black). The encoder does not store it, but it is recoverable
from the one-hot columns of x, so the parquet files do NOT need rebuilding.

lambda_decay: delta_t is left in plies, range 0..UNKNOWN_RECENCY (0..20). The
library's own default is 0.1. Measured on our data, 90.7% of edges sit at 20,
where 0.1 leaves a weight of 0.135: that squashes most of the board. Start at
0.02-0.05 and tune. Whatever you choose, it has to match the scale you feed:
pass time_divisor=20 and delta_t lands in [0,1], and lambda must grow by 20x
to have the same effect.


WHY THIS STREAMS INSTEAD OF LOADING EVERYTHING
-----------------------------------------------
One example is roughly 7 KB once it is a tensor. The training set has ~470k of
them, and the unbalanced one had 3.2M. ParquetGraphDataset is an
IterableDataset: it holds one parquet file at a time and drops it when it moves
on. It never deletes anything on disk.
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data

N_SQUARES = 64
N_EDGE_TYPES = 4            # attacks, defends, moves, pushes
N_PIECE_IDS = 13            # 0 = empty, 1-6 white pawn..king, 7-12 black
N_CLASSES = 4096            # from_square * 64 + to_square


"""
INPUT: x, the node feature matrix (64, F) as the encoder wrote it
OUTPUT: an int64 tensor (64,) with the piece id of every square, 0..12

NODE_FEATURES is ["pawn","knight","bishop","rook","queen","king","white",
"black","empty",...], so columns 0-5 are the piece type, 6-7 the colour and 8
the empty flag. The model's nn.Embedding wants one integer per node, so we fold
those flags back into a single id. Empty squares get 0, which is why the
embedding is sized 13 and not 12.
"""


def piece_ids_from_x(x):
    piece_type = np.argmax(x[:, 0:6], axis=1)      # 0..5
    is_black = x[:, 7] > 0.5
    occupied = x[:, 8] < 0.5                       # column 8 is the empty flag
    ids = np.where(occupied, 1 + piece_type + 6 * is_black, 0)
    return torch.tensor(ids, dtype=torch.long)


"""
INPUT: a row read back from parquet (a dict or a pandas Series), with_time
       (feed the real delta_t, or zeros to switch the decay off), time_divisor,
       keep_meta (carry fen/solution/puzzle_id, needed only at eval time)
OUTPUT: a torch_geometric.data.Data shaped for DualGATTimeAwareETModel

The parquet stores nested lists, because pyarrow cannot hold a 2-D array.
"""


def to_pyg_data(row, with_time=True, time_divisor=1.0, keep_meta=False):
    x_np = np.asarray(row["x"].tolist(), dtype=np.float32)
    edge_index = torch.tensor(
        np.asarray(row["edge_index"].tolist(), dtype=np.int64))
    one_hot = np.asarray(row["edge_attr"].tolist(), dtype=np.float32)

    # one-hot -> index, because nn.Embedding takes indices
    edge_type = torch.tensor(np.argmax(one_hot, axis=1), dtype=torch.long)

    n_edges = edge_index.shape[1]
    if with_time:
        if row["edge_time"] is None:
            raise ValueError(
                "with_time=True but this parquet carries no edge_time: it was "
                "built without --timing. Rebuild it, or pass with_time=False.")
        t = np.asarray(row["edge_time"], dtype=np.float32) / time_divisor
        time = torch.tensor(t)
    else:
        # zeros mean exp(-lambda * 0) = 1 on every edge: the decay is present
        # but does nothing. That is what makes the ablation clean, because the
        # two models stay architecturally identical.
        time = torch.zeros(n_edges, dtype=torch.float32)

    legal = torch.tensor(np.asarray(row["legal_moves"], dtype=np.int64))

    data = Data(
        x=torch.tensor(x_np),
        event_ids=piece_ids_from_x(x_np),
        edge_index=edge_index,
        edge_type=edge_type,
        time=time,
        y=torch.tensor([int(row["y"])], dtype=torch.long),
        # legal_moves has a different length in every position, so the count
        # has to travel with it or the batch cannot be taken apart again
        legal_moves=legal,
        n_legal=torch.tensor([len(legal)], dtype=torch.long),
        n_remaining=torch.tensor([int(row["n_remaining"])], dtype=torch.long),
        puzzle_n=torch.tensor([int(row["puzzle_n"])], dtype=torch.long),
    )
    if keep_meta:
        # plain python objects: the DataLoader collates them into a list.
        # Only needed to replay the line at eval time, so it is off by default.
        data.puzzle_id = row["puzzle_id"]
        data.fen = row["fen"]
        data.solution = list(row["solution"])
    return data


"""
INPUT: a batch produced by torch_geometric.loader.DataLoader
OUTPUT: a (B, 4096) tensor, 0.0 on legal moves and -inf everywhere else

The model has 4096 output classes but only ~26 are legal in any position. Add
this to the logits before argmax or softmax.
"""


def legal_mask(batch, n_classes=N_CLASSES):
    b = batch.n_legal.shape[0]
    mask = torch.full((b, n_classes), float("-inf"))
    offset = 0
    for i, k in enumerate(batch.n_legal.tolist()):
        mask[i, batch.legal_moves[offset:offset + k]] = 0.0
        offset += k
    return mask


class ParquetGraphDataset(IterableDataset):
    """
    Streams the parquet shards written by db_rows_to_graphs.py.

    Shuffling happens on two levels, because reading rows in random order
    across files would mean re-reading a file per example: the file order is
    shuffled, and the rows inside each loaded file are shuffled. Good enough
    for SGD, and it keeps exactly one file in memory.
    """

    def __init__(self, path, with_time=True, time_divisor=1.0,
                 keep_meta=False, shuffle=True, seed=0):
        super().__init__()
        if os.path.isdir(path):
            self.files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        else:
            self.files = [path]
        if not self.files:
            raise FileNotFoundError(f"no parquet file under {path}")
        self.with_time = with_time
        self.time_divisor = time_divisor
        self.keep_meta = keep_meta
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        import pandas as pd

        files = list(self.files)
        # with num_workers > 0 every worker would replay the whole dataset:
        # give each one its own slice of the files
        info = get_worker_info()
        if info is not None:
            files = files[info.id::info.num_workers]

        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(files)

        for f in files:
            df = pd.read_parquet(f)
            # to_dict is ~2.5x faster per row than repeated df.iloc[i]
            records = df.to_dict("records")
            order = list(range(len(records)))
            if self.shuffle:
                random.Random(self.seed + self.epoch + hash(f)).shuffle(order)
            for i in order:
                yield to_pyg_data(records[i], self.with_time,
                                  self.time_divisor, self.keep_meta)

    def set_epoch(self, epoch):
        """Call this between epochs or every epoch sees the same order."""
        self.epoch = epoch
