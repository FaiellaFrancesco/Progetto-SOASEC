"""
graph_io.py

Parquet cannot store 2-D arrays: pyarrow raises
    ArrowInvalid: Can only convert 1-dimensional array values ...
so x (64, F), edge_index (2, E) and edge_attr (E, T) are flattened before being
written and reshaped after being read.

flatten_example and rebuild_example are a PAIR. If one changes and the other
does not, the arrays come back with the wrong shape and nothing complains:
keep them in the same file and change them together.
"""

import numpy as np

N_SQUARES = 64


def flatten_example(e):
    """One example from row_to_graph -> a dict parquet can store."""
    r = dict(e)
    r["n_features"] = int(e["x"].shape[1])          # 12, or 13 with timing
    r["n_edge_types"] = int(e["edge_attr"].shape[1])
    r["x"] = e["x"].reshape(-1)
    r["edge_index"] = e["edge_index"].reshape(-1)
    r["edge_attr"] = e["edge_attr"].reshape(-1)
    return r


def rebuild_example(r):
    """A row read back from parquet -> the arrays with their original shapes."""
    e = dict(r)
    e["x"] = np.asarray(r["x"], dtype=np.float32).reshape(
        N_SQUARES, int(r["n_features"]))
    e["edge_index"] = np.asarray(r["edge_index"], dtype=np.int64).reshape(2, -1)
    e["edge_attr"] = np.asarray(r["edge_attr"], dtype=np.float32).reshape(
        -1, int(r["n_edge_types"]))
    if r.get("edge_time") is not None:
        e["edge_time"] = np.asarray(r["edge_time"], dtype=np.float32)
    e["legal_moves"] = np.asarray(r["legal_moves"], dtype=np.int16)
    return e
