"""
bridge between db_rows_to_graph.py and the library TimeGNN

#important

TimeAwareGATConv reads the time from the FIRST COLUMN of edge_attr:
 
    def edge_attention(self, alpha, edge_attr):
        time_diff = edge_attr[:, 0]                 # <-- column 0
        decay = torch.exp(-self.lambda_decay * time_diff)
        return alpha * decay.unsqueeze(-1)
"""
import glob
import os
import random
 
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data
 
N_SQUARES = 64
N_EDGE_TYPES = 4

"""
INPUT: Una riga letta del parquet
OUTPUT: a torch_geometric.data.Data (tensori)

praticamente entrano liste di liste ed escono gli stessi dati con formato giusto.

!! edge_attr passa a 5 colonne, viene aggiunto il time va davanti il resto è il one hot originale.

""" 
def to_pyg_data(row, with_time=True, time_divisor=1.0, keep_meta=False):
    x = torch.tensor(np.asarray(row["x"].tolist(), dtype=np.float32))
    edge_index = torch.tensor(
        np.asarray(row["edge_index"].tolist(), dtype=np.int64))
    edge_type = torch.tensor(
        np.asarray(row["edge_attr"].tolist(), dtype=np.float32))
 
    if with_time:
        if row["edge_time"] is None:
            raise ValueError(
                "with_time=True but this parquet has no edge_time: it was "
                "built without --timing. Rebuild it, or pass with_time=False.")
        t = torch.tensor(np.asarray(row["edge_time"], dtype=np.float32))
        t = (t / time_divisor).unsqueeze(1)
        # delta_t FIRST: TimeAwareGATConv reads edge_attr[:, 0]
        edge_attr = torch.cat([t, edge_type], dim=1)
    else:
        edge_attr = edge_type
 
    legal = torch.tensor(np.asarray(row["legal_moves"], dtype=np.int64))
 
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([int(row["y"])], dtype=torch.long),
        # legal_moves has a different length in every position, so the number
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
 
The model has 4096 output classes but almost all of them are illegal in any
given position. Add this to the logits before argmax or softmax.
"""
 
 
def legal_mask(batch, n_classes=4096):
    b = batch.n_legal.shape[0]
    mask = torch.full((b, n_classes), float("-inf"))
    # legal_moves arrives as one long concatenated vector: split it back
    # using the per-graph counts
    offset = 0
    for i, k in enumerate(batch.n_legal.tolist()):
        mask[i, batch.legal_moves[offset:offset + k]] = 0.0
        offset += k
    return mask
 
 
"""
INPUT: la cartella dei parquet (o un singolo file .parquet)
OUTPUT: un iteratore che restituisce un oggetto Data alla volta

Tiene in memoria un file parquet alla volta e lo scarta quando passa al successivo.
I file su disco non vengono mai toccati.
"""
class ParquetGraphDataset(IterableDataset):
   
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
            order = list(range(len(df)))
            if self.shuffle:
                random.Random(self.seed + self.epoch + hash(f)).shuffle(order)
            for i in order:
                yield to_pyg_data(df.iloc[i], self.with_time,
                                  self.time_divisor, self.keep_meta)
 
    def set_epoch(self, epoch):
        self.epoch = epoch
