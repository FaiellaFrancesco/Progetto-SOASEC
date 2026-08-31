from fen_to_graph import *
from graph_io import flatten_example
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from tqdm import tqdm
import pandas as pd
import argparse
import os

CHUNK_SIZE = 256
OUT_SIZE = 2560
WORKERS = 8


def make_graph(chunk, unroll=True, use_timing=False):
    graphs = []
    dropped = 0                      # rows that produced no example at all
    for record in chunk.to_dict('records'):
        g = row_to_graph(record, use_timing, unroll)
        if g:
            graphs.extend(flatten_example(e) for e in g)
        else:
            dropped += 1
    return graphs, len(chunk), dropped


def make_db(
        input_path,
        output_path,
        chunk_size,
        out_size,
        max_workers,
        unroll,
        use_timing,
):
    # count total lines for progress bar
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f) - 1
    total_chunks = (total_lines + chunk_size - 1) // chunk_size

    df_iterator = pd.read_csv(input_path, chunksize=chunk_size)

    curr_batch = 0
    done = []

    rows_in = 0
    rows_dropped = 0
    examples_out = 0

    max_queue_size = max_workers * 2
    futures_queue = deque()

    pbar = tqdm(total=total_chunks, desc="Processing chunks")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for chunk in df_iterator:
            if len(futures_queue) >= max_queue_size:
                fut = futures_queue.popleft()
                res, n_rows, n_dropped = fut.result()
                rows_in += n_rows
                rows_dropped += n_dropped
                examples_out += len(res)
                pbar.update(1)

                for g in res:
                    if g:
                        done.append(g)

                while len(done) >= out_size:
                    out = done[:out_size]
                    out_path = os.path.join(
                        output_path, f"graphs_batch_{curr_batch:04d}.parquet")
                    pd.DataFrame(out).to_parquet(out_path, index=False)
                    done = done[out_size:]
                    curr_batch += 1

            fut = pool.submit(make_graph, chunk, unroll, use_timing)
            futures_queue.append(fut)

        while futures_queue:
            fut = futures_queue.popleft()
            res, n_rows, n_dropped = fut.result()
            rows_in += n_rows
            rows_dropped += n_dropped
            examples_out += len(res)
            pbar.update(1)

            for g in res:
                if g:
                    done.append(g)

            while len(done) >= out_size:
                out = done[:out_size]
                out_path = os.path.join(
                    output_path, f"graphs_batch_{curr_batch:04d}.parquet")
                pd.DataFrame(out).to_parquet(out_path, index=False)
                done = done[out_size:]
                curr_batch += 1

    pbar.close()

    if done:
        out_path = os.path.join(
            output_path, f"graphs_batch_{curr_batch:04d}.parquet")
        pd.DataFrame(done).to_parquet(out_path, index=False)
        curr_batch += 1

    kept = rows_in - rows_dropped
    print(f"\nrows read       : {rows_in:,}")
    print(f"rows dropped    : {rows_dropped:,}  ({rows_dropped / max(rows_in, 1):.2%})")
    print(f"rows kept       : {kept:,}")
    print(f"examples written: {examples_out:,}")
    if kept:
        print(f"examples per row: {examples_out / kept:.2f}")
    print(f"parquet batches : {curr_batch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_csv", type=str, required=True)
    parser.add_argument('-o', "--output_dir", type=str,
                        default="./processed_graphs")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--out_size", type=int, default=OUT_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    # the two encoder flags, so both datasets of the ablation can be built from
    # the same code: --timing on/off, --no-unroll for the test split
    parser.add_argument("--timing", action="store_true",
                        help="add think_time to the nodes and edge_time to the edges")
    parser.add_argument("--no-unroll", dest="unroll", action="store_false",
                        help="keep only the first solver move (test split)")
    parser.set_defaults(unroll=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    make_db(
        args.input_csv,
        args.output_dir,
        args.chunk_size,
        args.out_size,
        args.workers,
        unroll=args.unroll,
        use_timing=args.timing,
    )
