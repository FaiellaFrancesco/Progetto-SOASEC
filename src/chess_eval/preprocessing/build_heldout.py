# Run (build the verified held-out set):
#   python src/chess_eval/preprocessing/build_heldout.py \
#       --harvey-dir dataset/harvey \
#       --test-csv   dataset/test.csv \
#       --mates-csv  dataset/lichess_db_puzzle_mates_only.csv \
#       --train-csv  dataset/train.csv \
#       --out        dataset/heldout.csv
#
# Builds the held-out set of mate-in-n problems (n = 1..10) on which the GNN and
# the LLM are compared. Every problem is verified by Stockfish before it is kept.
#
# Sources (labelled in the `source` column):
#   classic       Bill Harvey's mates from historic master games (not Lichess,
#                 so they test out-of-distribution generalisation). n = 2,3,4.
#   lichess_test  our test split: never trained on, same distribution. n = 1..5.
#   lichess_long  Lichess mates of 6+ moves, all tagged "mateIn5", so our guard
#                 dropped them all -> the only source for n = 6..10.
#
# Verified = the reference line replays to checkmate AND Stockfish reports mate
# in EXACTLY n from the start (a shorter mate disqualifies it). Candidates are
# drawn at random per (n, source) bucket until it is full, so Stockfish never
# has to scan the whole test split.
#
# Output: one row per problem, always from the SOLVER's point of view. `fen` is
# the position with the solver to move (for Lichess the setup move is already
# played and kept in `setup_uci`, which the timed model needs; classic problems
# leave it empty); `solution_uci` is the reference line. The evaluation does not
# force the model to follow it -- Stockfish judges every move -- but it is kept
# for the report and the human-rated comparison.

import argparse
import os
import random
import re
import shutil
import sys
import time

import chess
import chess.engine
import pandas as pd

COLUMNS = ["id", "source", "n", "fen", "side", "setup_uci", "solution_uci",
           "solution_san", "rating", "header", "n_legal", "sf_mate", "sf_seconds"]


def position_key(fen):
    """The position part of a FEN (board, side, castling, en passant).

    The move counters are dropped: the same position at a different move number
    is still the same problem.
    """
    return " ".join(fen.split()[:4])


def check_line(board, moves):
    """Validate a line from `board`: return (uci list, san string), or None.

    None if any move is illegal or the line does not end in checkmate.
    """
    b = board.copy()
    for m in moves:
        if m not in b.legal_moves:
            return None
        b.push(m)
    if not b.is_checkmate():
        return None
    return [m.uci() for m in moves], board.variation_san(moves)


# --------------------------------------------------------------------------
# Source 1: Bill Harvey's files
# --------------------------------------------------------------------------

_MOVE_NUMBER = re.compile(r"^\d+\.+$|^\d+\.+(?=\S)")
_RESULT = {"1-0", "0-1", "1/2-1/2", "*"}


def parse_san_line(line, board):
    """Parse a Harvey-notation line (e.g. "1. hxg7+ Rxg7 2. Rh1#") into moves.

    Move numbers and results are skipped; check/mate/annotation marks are left
    to python-chess. Returns None if any token is not a legal SAN move.
    """
    b = board.copy()
    moves = []
    for tok in line.replace("…", "...").split():
        tok = _MOVE_NUMBER.sub("", tok)
        if not tok or tok in _RESULT:
            continue
        tok = tok.rstrip("!?")
        try:
            m = b.parse_san(tok)
        except ValueError:
            return None
        moves.append(m)
        b.push(m)
    return moves


def read_harvey(path, n):
    """Read one Harvey file (for mate-in-n): returns (candidates, n_unparsable).

    The files are blocks of three lines (header, FEN, solution). A line is taken
    as a FEN when python-chess accepts it; the non-empty line before it is the
    header, the one after it the solution.
    """
    lines = [l.strip() for l in open(path, encoding="utf-8", errors="replace")]
    out, bad = [], 0
    for i, line in enumerate(lines):
        if line.count("/") != 7:
            continue
        try:
            board = chess.Board(line)
        except ValueError:
            continue
        header = next((lines[j] for j in range(i - 1, -1, -1) if lines[j]), "")
        sol = next((lines[j] for j in range(i + 1, len(lines)) if lines[j]), "")
        moves = parse_san_line(sol, board)
        checked = check_line(board, moves) if moves else None
        n_solver = (len(moves) + 1) // 2 if moves else 0
        if checked is None or n_solver != n:
            bad += 1
            continue
        uci, san = checked
        out.append({"source": "classic", "n": n, "fen": board.fen(),
                    "setup_uci": "",
                    "solution_uci": " ".join(uci), "solution_san": san,
                    "rating": None, "header": header,
                    "id": f"harvey_n{n}_{len(out):04d}"})
    return out, bad


# --------------------------------------------------------------------------
# Sources 2 and 3: Lichess
# --------------------------------------------------------------------------

def read_lichess(path, source, n_min, n_max, exclude=None):
    """Filter a Lichess CSV into candidate rows grouped by n ({n: [row, ...]}), plus counters.

    A Lichess row starts one half-move early (FEN is before the opponent's move,
    Moves[0] is that move), so n = len(Moves)//2. Rows are only filtered here,
    not replayed -- the few actually used are replayed in verify() -- and
    `exclude` drops positions already seen in training.
    """
    groups, seen, skipped = {}, 0, 0
    cols = ["PuzzleId", "FEN", "Moves", "Rating", "GameUrl", "Themes"]
    head = pd.read_csv(path, nrows=0).columns
    cols = [c for c in cols if c in head]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=200_000):
        seen += len(chunk)
        n = (chunk["Moves"].str.count(" ") + 1) // 2
        ok = (n >= n_min) & (n <= n_max)
        if "Themes" in chunk.columns:
            # the full Lichess database also holds non-mate puzzles: without
            # this filter Stockfish would waste time rejecting them one by one
            ok &= chunk["Themes"].fillna("").str.contains("mate", case=False)
        keep = chunk[ok].copy()
        keep["n"] = n[keep.index]
        if exclude:
            pk = keep["FEN"].map(position_key)
            skipped += int(pk.isin(exclude).sum())
            keep = keep[~pk.isin(exclude)]
        for r in keep.itertuples(index=False):
            groups.setdefault(int(r.n), []).append(r)
    return groups, {"rows": seen, "excluded_as_seen_in_training": skipped}


def lichess_candidate(r, source):
    """Build a solver-POV candidate dict from a Lichess row, or None.

    None if the line is illegal or does not end in mate.
    """
    moves = r.Moves.split()
    board = chess.Board(r.FEN)
    try:
        board.push_uci(moves[0])
        line = [chess.Move.from_uci(u) for u in moves[1:]]
    except ValueError:
        return None
    checked = check_line(board, line)
    if checked is None:
        return None
    uci, san = checked
    return {"source": source, "n": int(r.n), "fen": board.fen(),
            "setup_uci": moves[0],
            "solution_uci": " ".join(uci), "solution_san": san,
            "rating": getattr(r, "Rating", None),
            "header": getattr(r, "GameUrl", ""), "id": r.PuzzleId}


# --------------------------------------------------------------------------
# Stockfish
# --------------------------------------------------------------------------

def verify(engine, c, seconds):
    """Fill a candidate with Stockfish's verdict, or None if not mate in n.

    Limit(mate=n) lets Stockfish stop as soon as it proves the mate; the time
    budget only matters when it cannot. A shorter mate than n disqualifies it.
    """
    board = chess.Board(c["fen"])
    t0 = time.time()
    info = engine.analyse(board, chess.engine.Limit(time=seconds, mate=c["n"]))
    took = time.time() - t0
    score = info.get("score")
    mate = score.pov(board.turn).mate() if score is not None else None
    if mate != c["n"]:
        return None
    c.update(sf_mate=mate, sf_seconds=round(took, 3),
             n_legal=board.legal_moves.count(),
             side="w" if board.turn == chess.WHITE else "b")
    return c


def budget(n, base):
    """Stockfish time budget (seconds) for a mate in n: deeper mates get more."""
    return base * max(1.0, n / 2)


def fill_bucket(pool, target, engine, rng, base, make):
    """Draw candidates in random order and verify until the bucket is full.

    Returns (accepted, tried, rejected).
    """
    order = list(range(len(pool)))
    rng.shuffle(order)
    accepted, tried, rejected = [], 0, 0
    for i in order:
        if len(accepted) >= target:
            break
        c = make(pool[i])
        tried += 1
        if c is None:
            rejected += 1
            continue
        c = verify(engine, c, budget(c["n"], base))
        if c is None:
            rejected += 1
            continue
        accepted.append(c)
    return accepted, tried, rejected


def parse_targets(text):
    out = {}
    for part in text.split(","):
        k, v = part.split(":")
        out[int(k)] = int(v)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harvey-dir", help="cartella con m8n2.txt, m8n3.txt, m8n4.txt")
    p.add_argument("--test-csv", help="il test split (n da 1 a 5)")
    p.add_argument("--mates-csv",
                   help="il file dei matti di Lichess prima dello split (n >= 6)")
    p.add_argument("--train-csv",
                   help="per escludere posizioni gia' viste in training")
    p.add_argument("--targets", default="1:30,2:30,3:30,4:30,5:30,"
                                        "6:20,7:20,8:15,9:10,10:10",
                   help="quanti problemi per ogni n")
    p.add_argument("--classic-share", type=float, default=0.5,
                   help="quota di problemi classici dove disponibili (n 2-4)")
    p.add_argument("--stockfish", default="stockfish")
    p.add_argument("--sf-time", type=float, default=1.0,
                   help="secondi di base per verificare un problema")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="heldout.csv")
    args = p.parse_args()

    # check the engine FIRST: reading the CSVs takes minutes, and finding out
    # only afterwards that Stockfish is missing wastes all of them
    if shutil.which(args.stockfish) is None and not os.path.isfile(args.stockfish):
        sys.exit(f"Stockfish non trovato ('{args.stockfish}').\n"
                 f"  Mac:   brew install stockfish\n"
                 f"  Colab: !apt-get install -y stockfish  e poi "
                 f"--stockfish /usr/games/stockfish")

    targets = parse_targets(args.targets)
    rng = random.Random(args.seed)

    # ---- pools -------------------------------------------------------------
    pools = {}                         # (n, source) -> (rows, make function)

    if args.harvey_dir:
        print("== problemi classici (Bill Harvey)")
        for n in (2, 3, 4):
            path = os.path.join(args.harvey_dir, f"m8n{n}.txt")
            if not os.path.isfile(path):
                print(f"  manca {path}: salto n={n}")
                continue
            rows, bad = read_harvey(path, n)
            print(f"  n={n}: {len(rows)} leggibili, {bad} scartati in lettura")
            pools[(n, "classic")] = (rows, lambda c: dict(c))

    exclude = None
    if args.train_csv:
        print("== posizioni del training, da escludere")
        exclude = set(pd.read_csv(args.train_csv, usecols=["FEN"])["FEN"]
                      .map(position_key))
        print(f"  {len(exclude):,} posizioni")

    if args.test_csv:
        print("== test split di Lichess (n 1-5)")
        groups, cnt = read_lichess(args.test_csv, "lichess_test", 1, 5, exclude)
        print(f"  {cnt['rows']:,} righe, {cnt['excluded_as_seen_in_training']} "
              f"escluse perche' presenti nel training")
        for n, rows in sorted(groups.items()):
            print(f"  n={n}: {len(rows):,} candidati")
            pools[(n, "lichess_test")] = (
                rows, lambda r: lichess_candidate(r, "lichess_test"))

    if args.mates_csv:
        print("== matti lunghi di Lichess (n 6-10)")
        groups, cnt = read_lichess(args.mates_csv, "lichess_long", 6, 10, exclude)
        print(f"  {cnt['rows']:,} righe lette")
        for n, rows in sorted(groups.items()):
            print(f"  n={n}: {len(rows):,} candidati")
            pools[(n, "lichess_long")] = (
                rows, lambda r: lichess_candidate(r, "lichess_long"))

    if not pools:
        sys.exit("nessuna sorgente: indica almeno una fra --harvey-dir, "
                 "--test-csv, --mates-csv")

    # ---- per n: how many from each source ---------------------------------
    plan = []
    for n, total in sorted(targets.items()):
        sources = [s for (m, s) in pools if m == n]
        if not sources:
            print(f"!! n={n}: nessuna sorgente disponibile")
            continue
        if "classic" in sources and len(sources) > 1:
            k = round(total * args.classic_share)
            plan += [(n, "classic", k)] + \
                    [(n, s, total - k) for s in sources if s != "classic"]
        else:
            plan += [(n, sources[0], total)]

    # ---- verification -----------------------------------------------------
    print(f"\n== verifica con Stockfish ({args.stockfish})")
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({"Threads": args.threads, "Hash": 256})
    chosen = []
    try:
        for n, source, k in plan:
            rows, make = pools[(n, source)]
            t0 = time.time()
            got, tried, rejected = fill_bucket(rows, k, engine, rng,
                                               args.sf_time, make)
            flag = "" if len(got) == k else f"   !! solo {len(got)} su {k}"
            print(f"  n={n:<2} {source:<13} {len(got):>3}/{k:<3} "
                  f"(provati {tried}, scartati {rejected}, "
                  f"{time.time() - t0:.0f} s){flag}")
            chosen += got
    finally:
        engine.quit()

    df = pd.DataFrame(chosen)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COLUMNS].sort_values(["n", "source", "id"]).reset_index(drop=True)
    df.to_csv(args.out, index=False)

    print(f"\n== {len(df)} problemi scritti in {args.out}")
    print(df.groupby(["n", "source"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()