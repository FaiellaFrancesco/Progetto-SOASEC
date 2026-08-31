from fen_to_graph import *
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from tqdm import tqdm
import pandas as pd
import argparse
import os
import pickle
import glob

CHUNK_SIZE = 256
OUT_SIZE = 2560
WORKERS = 8


def make_graph(chunk, unroll=True, use_timing=False):
    graphs = []
    for record in chunk.to_dict('records'):
        g = row_to_graph(record, use_timing, unroll)
        if g:
            # Needed to use pyarrow to handle parquet shards
            for item in g:
                for k, v in item.items():
                    if isinstance(v, np.ndarray):
                        item[k] = v.tolist()
            graphs.extend(g)
    return graphs


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

    max_queue_size = max_workers * 2
    futures_queue = deque()

    pbar = tqdm(total=total_chunks, desc="Processing chunks")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for chunk in df_iterator:
            # if the queue is full we wait on the oldest chunk to finish
            if len(futures_queue) >= max_queue_size:
                fut = futures_queue.popleft()
                res = fut.result()
                pbar.update(1)

                for g in res:
                    if g:
                        done.append(g)
                while len(done) >= out_size:
                    out = done[:out_size]
                    out_path = os.path.join(output_path, f"graphs_batch_{
                                            curr_batch:04d}.parquet")
                    tmp = pd.DataFrame(out)
                    tmp.to_parquet(out_path, index=False, engine='pyarrow')
                    done = done[out_size:]
                    curr_batch += 1

            fut = pool.submit(make_graph, chunk, unroll, use_timing)
            futures_queue.append(fut)

        while futures_queue:
            fut = futures_queue.popleft()
            res = fut.result()
            pbar.update(1)

            for g in res:
                if g:
                    done.append(g)

            while len(done) >= out_size:
                out = done[:out_size]
                out_path = os.path.join(output_path, f"graphs_batch_{
                                        curr_batch:04d}.parquet")
                temp = pd.DataFrame(out).to_parquet(
                    out_path, index=False, engine='pyarrow')
                done = done[out_size:]
                curr_batch += 1

    pbar.close()

    # save any leftover records
    if done:
        out_path = os.path.join(output_path, f"graphs_batch_{
                                curr_batch:04d}.parquet")
        pd.DataFrame(out).to_parquet(out_path, index=False, engine='pyarrow')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_csv", type=str, required=True)
    parser.add_argument('-o', "--output_dir", type=str,
                        default="./processed_graphs")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--out_size", type=int, default=OUT_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    make_db(
        args.input_csv,
        args.output_dir,
        args.chunk_size,
        args.out_size,
        args.workers,
        unroll=True,
        use_timing=True
    )
