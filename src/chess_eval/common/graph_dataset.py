# Library module (imported, not run directly).
# Used by: training.train, evaluation.*, diagnostics.*, tests.*
#
# Streams the parquet graph shards (written by preprocessing.db_rows_to_graphs)
# as torch_geometric Data objects for DualGATTimeAwareETModel.
#
#   from chess_eval.common.graph_dataset import ParquetGraphDataset
#   from torch_geometric.loader import DataLoader
#   ds = ParquetGraphDataset("./graphs/train_timing", with_time=True)
#   for batch in DataLoader(ds, batch_size=64):
#       out = model(batch)
#
# Contract with the model (read off TimeGNN's source): it uses edge_index,
# edge_type (an INT index into an nn.Embedding, not a one-hot), time (a SEPARATE
# field, not a column of edge_attr), event_ids and x. It builds edge_attr on its
# own, so we never pass it; our encoder writes a 4-col one-hot, so we argmax it
# back to an index.
#
# lambda_decay: delta_t is in plies, range 0..20. ~91% of edges sit at 20, so
# the library default 0.1 squashes the board (weight 0.135); start at 0.02-0.05.
# time_divisor=20 maps delta_t to [0,1], and then lambda must grow 20x to match.
#
# It streams (IterableDataset) one parquet file at a time instead of loading
# everything: ~470k examples at ~7 KB each would not fit comfortably in memory.

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


def piece_ids_from_x(x):
    """Per-square piece id 0..12 (0 empty, 1-6 white pawn..king, 7-12 black).

    The model's nn.Embedding wants one integer per node, so fold the one-hot
    type/colour/empty columns of x back into a single id. Empty squares get 0,
    which is why the embedding is sized 13, not 12.
    """
    piece_type = np.argmax(x[:, 0:6], axis=1)      # 0..5
    is_black = x[:, 7] > 0.5
    occupied = x[:, 8] < 0.5                       # column 8 is the empty flag
    ids = np.where(occupied, 1 + piece_type + 6 * is_black, 0)
    return torch.tensor(ids, dtype=torch.long)


def to_pyg_data(row, with_time=True, time_divisor=1.0, keep_meta=False):
    """Build a torch_geometric Data from a parquet row, shaped for the model.

    with_time feeds the real delta_t (zeros switch the decay off but keep the
    two models architecturally identical); keep_meta carries fen/solution/
    puzzle_id, needed only at eval time. The parquet stores nested lists because
    pyarrow cannot hold a 2-D array.
    """
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
        # zeros -> exp(-lambda*0) = 1 on every edge: decay present but inert
        time = torch.zeros(n_edges, dtype=torch.float32)

    legal = torch.tensor(np.asarray(row["legal_moves"], dtype=np.int64))

    data = Data(
        x=torch.tensor(x_np),
        event_ids=piece_ids_from_x(x_np),
        edge_index=edge_index,
        edge_type=edge_type,
        time=time,
        y=torch.tensor([int(row["y"])], dtype=torch.long),
        # legal_moves has a different length per position, so its count must
        # travel with it or the batch cannot be split apart again
        legal_moves=legal,
        n_legal=torch.tensor([len(legal)], dtype=torch.long),
        n_remaining=torch.tensor([int(row["n_remaining"])], dtype=torch.long),
        puzzle_n=torch.tensor([int(row["puzzle_n"])], dtype=torch.long),
    )
    if keep_meta:
        # plain python objects, collated into a list by the DataLoader; only
        # needed to replay the line at eval time, so off by default
        data.puzzle_id = row["puzzle_id"]
        data.fen = row["fen"]
        data.solution = list(row["solution"])
    return data


def legal_mask(batch, n_classes=N_CLASSES):
    """Additive mask (B, 4096): 0.0 on legal moves, -inf elsewhere.

    Add it to the logits before argmax/softmax. The model has 4096 classes but
    only ~26 are legal in any position.
    """
    b = batch.n_legal.shape[0]
    mask = torch.full((b, n_classes), float("-inf"))
    offset = 0
    for i, k in enumerate(batch.n_legal.tolist()):
        mask[i, batch.legal_moves[offset:offset + k]] = 0.0
        offset += k
    return mask


class ParquetGraphDataset(IterableDataset):
    """Streams the parquet shards written by preprocessing.db_rows_to_graphs.

    Shuffling is two-level (file order, then rows within the loaded file),
    because reading rows in global random order would re-read a file per
    example. Good enough for SGD and keeps exactly one file in memory. Nothing
    is deleted on disk.
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
        # with num_workers > 0, give each worker its own slice of the files so
        # they don't all replay the whole dataset
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
        """Call between epochs, or every epoch sees the same order."""
        self.epoch = epoch