import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, List, Dict, Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch_geometric.data import Data, Batch


def record_to_pyg_data(record: Dict[str, Any]) -> Data:
    """Converts a parquet record row into a PyG Data object."""
    x = torch.tensor(record["x"], dtype=torch.float32)
    edge_index = torch.tensor(record["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(record["edge_attr"], dtype=torch.float32)
    y = torch.tensor([record["y"]], dtype=torch.long)

    edge_time = None
    if record.get("edge_time") is not None:
        edge_time = torch.tensor(record["edge_time"], dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_time=edge_time,
        y=y,
        legal_moves=np.array(record["legal_moves"], dtype=np.int16),
        puzzle_id=record.get("puzzle_id"),
        fen=record.get("fen"),
        rating=record.get("rating"),
        think_time=record.get("think_time"),
    )
    return data


class ParquetDbLoader:
    def __init__(
        self,
        parquet_file: str,
        batch_size: int = 64,
        num_workers: int = 4,
        max_prefetch_batches: int = 16,
    ):
        self.parquet_file = parquet_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_queue = queue.Queue(maxsize=max_prefetch_batches)
        self._sentinel = object()

        # Check total row groups for threaded reads
        parquet_meta = pq.ParquetFile(self.parquet_file)
        self.num_row_groups = parquet_meta.num_row_groups

    def _read_row_group(self, group_idx: int) -> List[Data]:
        pf = pq.ParquetFile(self.parquet_file)
        table = pf.read_row_group(group_idx)
        records = table.to_pylist()
        return [record_to_pyg_data(r) for r in records]

    def _producer(self):
        accumulator: List[Data] = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for group_graphs in executor.map(self._read_row_group, range(self.num_row_groups)):
                accumulator.extend(group_graphs)

                while len(accumulator) >= self.batch_size:
                    batch_list = accumulator[: self.batch_size]
                    accumulator = accumulator[self.batch_size:]
                    self.batch_queue.put(Batch.from_data_list(batch_list))

        if accumulator:
            self.batch_queue.put(Batch.from_data_list(accumulator))

        self.batch_queue.put(self._sentinel)

    def __iter__(self) -> Generator[Batch, None, None]:
        producer_thread = threading.Thread(target=self._producer, daemon=True)
        producer_thread.start()

        while True:
            batch = self.batch_queue.get()
            if batch is self._sentinel:
                break
            yield batch
