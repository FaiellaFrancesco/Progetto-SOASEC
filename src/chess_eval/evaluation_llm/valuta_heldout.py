# Run (play one player, then build the report):
#   # play: one run per player, appended to a JSONL
#   python src/chess_eval/evaluation_llm/valuta_heldout.py play --player stockfish \
#       --heldout dataset/heldout.csv --out risultati/heldout/stockfish.jsonl
#   python src/chess_eval/evaluation_llm/valuta_heldout.py play --player random \
#       --heldout dataset/heldout.csv --out risultati/heldout/random.jsonl
#   python src/chess_eval/evaluation_llm/valuta_heldout.py play --player gnn \
#       --ckpt ckpt_fix/plain/best.pt --heldout dataset/heldout.csv \
#       --out risultati/heldout/gnn_plain.jsonl
#   python src/chess_eval/evaluation_llm/valuta_heldout.py play --player llm --backend openai \
#       --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B \
#       --heldout dataset/heldout.csv --out risultati/heldout/qwen.jsonl
#   # report: tables per n and per source, McNemar between every pair
#   python src/chess_eval/evaluation_llm/valuta_heldout.py report \
#       risultati/heldout/*.jsonl --out risultati/heldout/report.json
#
# The referee of the held-out comparison: a player (gnn, llm, or a control
# player) tries to solve every mate-in-n problem in heldout.csv while Stockfish
# defends and judges. The same rules apply to every player, so results compare
# puzzle by puzzle.
#
# Rules: a mate in n is SOLVED if the player delivers checkmate within n of its
# own moves against Stockfish's defence. Each turn: the player proposes a move
# (illegal/missing/unreadable = failed); mate = solved (even if faster than n),
# no mate on the n-th move = failed; Stockfish plays its best (longest-resisting)
# defence and then judges whether a forced mate still exists in the moves left --
# if not, the player threw the win away and fails here, without spending more
# calls (which matters for the LLM, where each call costs minutes). Any winning
# line counts, not only the one in the CSV. ok_moves counts how many of the
# player's moves kept the forced mate, so ok_moves >= 1 is the first-move
# accuracy, comparable with evaluate_models.py.
#
# Players: stockfish (sanity check -- MUST score 100%, else the judge/defence is
# wrong), random (the floor every model must beat), gnn (our model via the same
# graph encoder as training), llm (backends: openai = any OpenAI-compatible
# server such as vLLM/llama.cpp/Ollama/a hosted API, hf = transformers in
# process, mock = random moves + garbage, only to test this script).
#
# Lockstep: all puzzles advance together, one move per round (every puzzle's
# move 1, then move 2, ...). One round is one batch for the GNN and one batch of
# parallel requests for the LLM, which is what makes the LLM affordable.
#
# Checking the referee, before trusting any number: (1) the stockfish player
# must score 100%; (2) `recheck` re-searches every "matto perso" verdict with a
# much larger budget (50M nodes) and must overturn none:
#     python src/chess_eval/evaluation_llm/valuta_heldout.py recheck \
#         risultati/heldout/gnn_plain.jsonl --heldout dataset/heldout.csv
#
# Resuming: each finished puzzle is appended to the JSONL as it is done. Re-run
# the same command to resume -- puzzles already in the file are skipped, half-way
# ones restart from the beginning.

import argparse
import json
import math
import os
import random
import queue
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import chess
import chess.engine
import pandas as pd


# ==========================================================================
# the state of one problem while it is being played
# ==========================================================================

class Game:
    """One held-out problem, plus everything that happened to it."""

    def __init__(self, row):
        self.id = str(row["id"])
        self.source = row["source"]
        self.n = int(row["n"])
        self.fen = row["fen"]
        self.rating = (float(row["rating"]) if pd.notna(row.get("rating"))
                       else 1500.0)
        self.reference = str(row["solution_uci"]).split()
        setup = row.get("setup_uci")
        self.setup = setup if isinstance(setup, str) and setup else None

        self.board = chess.Board(self.fen)
        self.solver = self.board.turn
        self.k = 0               # moves the player has made
        self.ok_moves = 0        # of which kept a forced mate
        self.moves = []          # every move played, both sides, in UCI
        self.done = False
        self.solved = False
        self.reason = ""
        self.seconds = 0.0       # player's thinking time charged to this game
        self.tokens = 0
        self.calls = 0
        self.responses = []
        self.plan = None         # line mode of the LLM: the planned moves

        # move history for the GNN with timing, built exactly as in
        # playout_models.Puzzle.reset: the opponent's setup move, when there
        # is one, is ply 1. Classic problems have no history at all.
        self.played = 0
        self.last_moved = {}
        if self.setup:
            self.played = 1
            self._remember(chess.Move.from_uci(self.setup))

    def _remember(self, move):
        # same two lines as fen_to_graph.update_last_moved
        self.last_moved.pop(move.from_square, None)
        self.last_moved[move.to_square] = self.played

    def push(self, move):
        self.board.push(move)
        self.moves.append(move.uci())
        self.played += 1
        self._remember(move)

    @property
    def remaining(self):
        return self.n - self.k

    def finish(self, solved, reason):
        self.done, self.solved, self.reason = True, solved, reason

    def record(self, player):
        return {"id": self.id, "source": self.source, "n": self.n,
                "player": player, "solved": self.solved, "reason": self.reason,
                "moves_made": self.k, "ok_moves": self.ok_moves,
                "first_ok": self.ok_moves >= 1 or (self.solved and self.k == 1),
                "moves": " ".join(self.moves),
                "reference": " ".join(self.reference),
                "seconds": round(self.seconds, 3),
                "llm_calls": self.calls, "tokens": self.tokens,
                "responses": self.responses}


def load_games(path, limit=None, n_max=None, ids=None, per_n=None, seed=0):
    df = pd.read_csv(path)
    if n_max:
        df = df[df["n"] <= n_max]
    if ids:
        df = df[df["id"].astype(str).isin(ids)]
    if per_n:
        # a pilot: the same K problems per band of n for every model, so the
        # pilots of different models are comparable. Fixed seed: a relaunch
        # picks the same ones and resumes instead of starting over
        bands = df["n"].map(band)
        parts = [df[bands == b].sample(min(per_n, int((bands == b).sum())),
                                       random_state=seed)
                 for b in sorted(set(bands))]
        df = pd.concat(parts).sort_values(["n", "id"])
    if limit:
        df = df.head(limit)
    return [Game(r) for r in df.to_dict("records")]


# ==========================================================================
# the referee: Stockfish defends and judges
# ==========================================================================

class Referee:
    """
    Stockfish as defender and judge.

    Reproducibility: Threads=1, a node limit instead of a time limit, and an
    EMPTY hash table before every search. The last one matters more than it
    looks: Stockfish's search depends on what is left in its hash from the
    previous search, and we measured the same position going from "no mate
    found in 2M nodes" (hash full of the defence search) to "mate in 7 found
    in 0.5M nodes" (empty hash). With the hash cleared every time, the same
    position always gets the same verdict, whatever was searched before and
    whichever engine process does it.

    Speed: `workers` Stockfish processes judge different games at the same
    time. A failed move is the expensive case (proving there is no mate means
    searching the whole budget), and a weak player fails a lot.
    """

    def __init__(self, path, defence_nodes, judge_nodes, defence="stockfish",
                 escalate_nodes=0, workers=1):
        self.pool = queue.Queue()
        self.engines = []
        for _ in range(max(1, workers)):
            e = chess.engine.SimpleEngine.popen_uci(path)
            e.configure({"Threads": 1, "Hash": 64})
            self.engines.append(e)
            self.pool.put(e)
        self.workers = len(self.engines)
        self.defence_nodes = defence_nodes
        self.judge_nodes = judge_nodes
        self.escalate_nodes = escalate_nodes
        self.escalations = 0
        self.lock = threading.Lock()
        self.defence = defence

    def close(self):
        for e in self.engines:
            e.quit()

    def _search(self, board, limit):
        """One search on a free engine, starting from an empty hash."""
        e = self.pool.get()
        try:
            # a new `game` object makes python-chess send ucinewgame, which
            # clears Stockfish's hash table
            return e.analyse(board, limit, game=object())
        finally:
            self.pool.put(e)

    def defend(self, g):
        """The defender's move and `proved` (True if the defence search already
        proved the player still mates in time, so the judge can skip it).

        In "reference" mode the recorded reply is used while the player follows
        the recorded line exactly; once it deviates, Stockfish takes over.
        """
        if self.defence == "reference":
            i = len(g.moves)
            if g.moves == g.reference[:i] and i < len(g.reference):
                return chess.Move.from_uci(g.reference[i]), False
        info = self._search(g.board, chess.engine.Limit(nodes=self.defence_nodes))
        # after the reply the player has `remaining` moves left
        return info["pv"][0], self._proved(info, g)

    def still_mates(self, g):
        """True if Stockfish proves a mate for the player within the moves it has left.

        Limit(mate=r) stops as soon as the mate is proved, so a good move costs
        milliseconds; only a move that lost the mate runs to the node limit.
        """
        info = self._search(g.board, chess.engine.Limit(
            mate=g.remaining, nodes=self.judge_nodes))
        if self._proved(info, g):
            return True
        # Not proved. Usually the move really was bad, but a DEEP mate can
        # need far more nodes: the first version of this judge wrongly failed
        # a correct mate in 7 that took 12.7M nodes. So a second, 10x longer
        # search - but only when the position still looks won (Stockfish
        # sees a mate, just a longer one, or the player is at least 3 pawns
        # up). A blunder that throws the advantage away is not searched again.
        score = info.get("score")
        if self.escalate_nodes <= self.judge_nodes or score is None:
            return False
        s = score.pov(g.solver)
        promising = (s.mate() is not None and s.mate() > 0) or \
                    (s.score() is not None and s.score() >= 300)
        if not promising:
            return False
        with self.lock:
            self.escalations += 1
        info = self._search(g.board, chess.engine.Limit(
            mate=g.remaining, nodes=self.escalate_nodes))
        return self._proved(info, g)

    @staticmethod
    def _proved(info, g):
        score = info.get("score")
        mate = score.pov(g.solver).mate() if score is not None else None
        return mate is not None and 0 < mate <= g.remaining


def apply_turn(ref, g, move, why):
    """Advance the game by one full move (player + defence), or finish it.

    This is where the rules in the header live.
    """
    if move is None:
        g.finish(False, why or "nessuna mossa")
        return
    if move not in g.board.legal_moves:
        g.finish(False, "illegale")
        return
    g.push(move)
    g.k += 1
    if g.board.is_checkmate():
        g.ok_moves += 1
        g.finish(True, "matto")
        return
    if g.board.is_game_over():
        g.finish(False, "stallo o patta")
        return
    if g.remaining == 0:
        g.finish(False, "mosse finite senza matto")
        return
    reply, proved = ref.defend(g)
    g.push(reply)
    if not proved and not ref.still_mates(g):
        g.finish(False, "matto perso")
        return
    g.ok_moves += 1


def play_all(ref, player, games, out_path, name):
    """Run every game to the end, appending each to the JSONL as it finishes."""
    active = list(games)
    rnd = 0
    with open(out_path, "a") as out:
        while active:
            rnd += 1
            t0 = time.time()
            moves = player.choose(active)
            # one batched call answers every active game at once: its time is
            # split evenly among them. Stockfish's time is not counted: it is
            # the referee, not the player.
            share = (time.time() - t0) / len(active)
            for g in active:
                g.seconds += share
            # the games are independent: the referee judges them in parallel,
            # one Stockfish process each
            with ThreadPoolExecutor(ref.workers) as ex:
                list(ex.map(lambda gm: apply_turn(ref, gm[0], *gm[1]),
                            zip(active, moves)))
            for g in active:
                if g.done:
                    out.write(json.dumps(g.record(name)) + "\n")
            out.flush()
            os.fsync(out.fileno())
            done = sum(g.done for g in active)
            solved = sum(g.solved for g in active)
            print(f"  giro {rnd}: {len(active):>4} in gioco -> {solved} risolti, "
                  f"{done - solved} falliti   ({time.time() - t0:.0f} s)",
                  flush=True)
            active = [g for g in active if not g.done]


# ==========================================================================
# control players
# ==========================================================================

class StockfishPlayer:
    """Plays Stockfish's mating move. Must solve everything."""

    def __init__(self, path, nodes):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": 1, "Hash": 64})
        self.nodes = nodes

    def choose(self, games):
        return [(self.engine.play(g.board, chess.engine.Limit(
            mate=g.remaining, nodes=self.nodes), game=object()).move, "")
            for g in games]

    def close(self):
        self.engine.quit()


class RandomPlayer:
    """A random legal move. Seeded per game and per move: reproducible."""

    def __init__(self, seed):
        self.seed = seed

    def choose(self, games):
        out = []
        for g in games:
            rng = random.Random(f"{self.seed}-{g.id}-{g.k}")
            legal = sorted(g.board.legal_moves, key=lambda m: m.uci())
            out.append((rng.choice(legal), ""))
        return out

    def close(self):
        pass


# ==========================================================================
# the GNN
# ==========================================================================

class GNNPlayer:
    """
    Reuses the project's own code, so the graph it sees is built exactly as in
    training: load_model from evaluate_models, puzzle_to_data and the legal
    move mask from playout_models. This file must therefore sit in
    src/dataset_scripts, next to them (or point --code-dir there).

    The plain model does not know n. The timing model does (n_remaining goes
    into its think_time column): for n > 5 it receives values it has never
    seen in training, and that has to be said next to its results.
    """

    def __init__(self, ckpt, code_dir, batch_size):
        sys.path.insert(0, code_dir)
        try:
            import torch
            from torch_geometric.data import Batch
            from chess_eval.evaluation.evaluate_models import load_model
            from chess_eval.evaluation.playout_models import puzzle_to_data, batch_legal_mask
            from chess_eval.common.fen_to_graph import decode_move
        except ImportError as e:
            sys.exit(f"\nla GNN non si carica: {e}\n"
                     f"Serve l'ambiente Python del training (torch, "
                     f"torch_geometric e la libreria timegnn raggiungibile). "
                     f"Usa lo stesso con cui gira test_model_v2.py.\n"
                     f"Python in uso: {sys.executable}")

        self.torch, self.Batch = torch, Batch
        self.to_data, self.mask = puzzle_to_data, batch_legal_mask
        self.decode = decode_move
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, a, epoch, best = load_model(ckpt, self.device)
        self.with_time = bool(a["timing"])
        self.batch_size = batch_size
        print(f"GNN: {ckpt}  timing={self.with_time}  epoca {epoch}  "
              f"device {self.device}")
        if not (a.get("edge_inputs") and a.get("self_loops")):
            print("  ATTENZIONE: checkpoint senza edge_inputs/self_loops: "
                  "e' un modello vecchio (v1)")
        if self.with_time:
            print("  ATTENZIONE: modello timing, per n > 5 riceve un "
                  "n_remaining mai visto in training")

    def choose(self, games):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(games), self.batch_size):
                chunk = games[i:i + self.batch_size]
                # puzzle_to_data reads exactly these five attributes
                views = [SimpleNamespace(board=g.board, rating=g.rating,
                                         n_remaining=g.remaining,
                                         last_moved=g.last_moved,
                                         played=g.played) for g in chunk]
                batch = self.Batch.from_data_list(
                    [self.to_data(v, self.with_time) for v in views]
                ).to(self.device)
                logits = self.model(batch) + self.mask(batch, self.device)
                for g, y in zip(chunk, logits.argmax(dim=1).tolist()):
                    out.append((self.decode(int(y), g.board), ""))
        return out

    def close(self):
        pass


# ==========================================================================
# the LLM: prompt, parsing, backends
# ==========================================================================

SYSTEM = ("You are a strong chess player solving 'mate in n' puzzles.\n"
          "Method: list the forcing moves of the side to move (checks, "
          "captures, threats). For each candidate, calculate every reply of "
          "the opponent, not only the obvious one. Choose a move that forces "
          "checkmate within the stated number of moves against ANY defence.\n"
          "Coordinates: files a-h from White's left, ranks 1-8 from White's "
          "side. Only the moves in the legal-move list can be played.\n"
          "Answer with exactly one move in the requested format.")


def board_drawing(board):
    """ASCII board with coordinates, rank 8 on top: White upper case."""
    rows = []
    for rank in range(7, -1, -1):
        cells = []
        for f in range(8):
            p = board.piece_at(chess.square(f, rank))
            cells.append(p.symbol() if p else ".")
        rows.append(f"{rank + 1} " + " ".join(cells))
    rows.append("  a b c d e f g h")
    return "\n".join(rows)


def history_san(g):
    """The moves played so far in this puzzle, in SAN with move numbers."""
    if not g.moves:
        return "none"
    b = chess.Board(g.fen)
    return b.variation_san([chess.Move.from_uci(u) for u in g.moves])


def build_prompt(g, opts, mode, shots=()):
    """The chat messages for this game's current position (mode "move" or "line").

    Everything the GNN gets implicitly is made explicit: the position, whose
    turn it is, and the legal moves (so the LLM is judged on tactics, not on
    knowing the rules). Unlike the plain GNN, the LLM is also told n -- a human
    solver would be, and it costs nothing.
    """
    side = "White" if g.solver == chess.WHITE else "Black"
    r = g.remaining
    parts = [f"Position (FEN): {g.board.fen()}"]
    if opts.board:
        parts.append("Board (White = upper case, Black = lower case):\n"
                     + board_drawing(g.board))
    parts.append(f"{side} to move.")
    parts.append(f"Moves already played in this puzzle: {history_san(g)}")
    if mode == "line":
        parts.append(f"Task: {side} forces checkmate in {r} move"
                     f"{'s' if r > 1 else ''}. Give the whole winning line, "
                     f"{side}'s moves and the expected replies, in UCI.")
    else:
        parts.append(f"Task: {side} can force checkmate in at most {r} more "
                     f"move{'s' if r > 1 else ''}, against any defence. "
                     f"Find {side}'s best move now.")
    if opts.legal:
        # SAN marks a check with "+" and a mate with "#": left in, the list
        # would GIVE AWAY every mate in 1 (and the last move of every puzzle)
        # - the model would only have to look for the "#". The list must say
        # which moves are legal, nothing about what they achieve: the GNN does
        # not get that either. So the marks are stripped.
        legal = sorted(g.board.legal_moves, key=lambda m: m.uci())
        parts.append("Legal moves: " + ", ".join(
            f"{g.board.san(m).rstrip('+#')} ({m.uci()})" for m in legal))
    if mode == "line":
        parts.append("End your answer with one line in exactly this format:\n"
                     "LINE: <move> <move> <move> ...   (UCI, e.g. LINE: "
                     "e2e4 e7e5 d1h5)")
    else:
        parts.append("End your answer with one line in exactly this format:\n"
                     "MOVE: <move>   (UCI like e2e4, or SAN like Nf3)")
    msgs = ([{"role": "system", "content": SYSTEM}] + list(shots)
            + [{"role": "user", "content": "\n\n".join(parts)}])
    if getattr(opts, "system_in_user", False):
        # some models (Gemma through the Gemini API) have no system role:
        # the same text goes at the top of the first user message instead,
        # so every model still reads exactly the same words
        first = dict(msgs[1])
        first["content"] = SYSTEM + "\n\n" + first["content"]
        msgs = [first] + msgs[2:]
    return msgs


def example_answer(ex, mode):
    """The answer a good solver would write for one solved example.

    Generated from the solution line with python-chess, so it cannot contain a
    chess mistake: each move is labelled (check, capture, mate) and each reply
    marked as the only legal move or the best defence. It shows the model the
    behaviour we want -- calculate the whole line, answer with the first move
    only -- without hand-writing chess prose.
    """
    board = chess.Board(ex["fen"])
    line = [chess.Move.from_uci(u) for u in ex["solution_uci"].split()]
    side = "White" if board.turn == chess.WHITE else "Black"
    first_san = board.san(line[0])
    rows = []
    for i in range(0, len(line), 2):
        m = line[i]
        san = board.san(m)
        tags = []
        if board.is_capture(m):
            tags.append("capture")
        board.push(m)
        if board.is_checkmate():
            tags.append("checkmate")
        elif board.is_check():
            tags.append("check")
        row = f"Move {i // 2 + 1}: {san}" + (f" ({', '.join(tags)})" if tags else "")
        if i + 1 < len(line):
            reply = line[i + 1]
            only = board.legal_moves.count() == 1
            row += (f" - reply {board.san(reply)} "
                    f"({'the only legal reply' if only else 'best defence'})")
            board.push(reply)
        rows.append(row)
    n = (len(line) + 1) // 2
    first = line[0].uci()
    text = ("Forcing moves first: checks, captures, threats.\n"
            + "\n".join(rows)
            + f"\nThe line ends in checkmate in {n} move{'s' if n > 1 else ''} "
              f"for {side}.\n")
    if mode == "line":
        return text + "LINE: " + ex["solution_uci"]
    return text + f"So the move now is {first_san}.\nMOVE: {first}"


def load_examples(path, heldout_path):
    """
    INPUT: the few-shot file (or None), the held-out CSV
    OUTPUT: the examples, after checking none of them is a held-out problem

    An example taken from the held-out set would give the model the answer to
    a question it is then asked: the check makes that impossible by mistake.
    """
    if not path:
        return []
    with open(path) as f:
        examples = json.load(f)
    held = pd.read_csv(heldout_path)
    keys = set(held["fen"].map(lambda f: " ".join(f.split()[:4])))
    for ex in examples:
        if " ".join(ex["fen"].split()[:4]) in keys:
            sys.exit(f"l'esempio {ex['id']} e' un problema del held-out: "
                     f"rigenera il file degli esempi")
    return examples


def make_fewshot(train_csv, heldout_path, ns, rmin, rmax, seed, out_path):
    """Write the few-shot examples as JSON: one per requested n.

    Drawn at random from the TRAINING split (never val/test/held-out), in a
    middle rating band so they are typical rather than trivial or extreme. Each
    line is replayed and must end in checkmate.
    """
    held = pd.read_csv(heldout_path)
    held_keys = set(held["fen"].map(lambda f: " ".join(f.split()[:4])))
    rng = random.Random(seed)
    pools = {n: [] for n in ns}
    cols = [c for c in ("PuzzleId", "FEN", "Moves", "Rating", "Themes")
            if c in pd.read_csv(train_csv, nrows=0).columns]
    for chunk in pd.read_csv(train_csv, usecols=cols, chunksize=200_000):
        n = chunk["Moves"].str.split().str.len() // 2
        ok = n.isin(ns) & chunk["Rating"].between(rmin, rmax)
        for r, k in zip(chunk[ok].itertuples(index=False), n[ok]):
            pools[int(k)].append(r)
    examples = []
    for n in ns:
        pool = pools[n]
        rng.shuffle(pool)
        for r in pool:
            moves = r.Moves.split()
            board = chess.Board(r.FEN)
            try:
                board.push_uci(moves[0])
                fen = board.fen()
                for u in moves[1:]:
                    board.push_uci(u)
            except ValueError:
                continue
            if not board.is_checkmate() or \
                    " ".join(fen.split()[:4]) in held_keys:
                continue
            examples.append({"id": r.PuzzleId, "source": "train", "n": n,
                             "fen": fen, "setup_uci": moves[0],
                             "solution_uci": " ".join(moves[1:]),
                             "rating": int(r.Rating)})
            break
        else:
            sys.exit(f"nessun esempio valido con n={n} nella fascia "
                     f"{rmin}-{rmax}")
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=1)
    print(f"{len(examples)} esempi scritti in {out_path}:")
    for ex in examples:
        print(f"  n={ex['n']}  {ex['id']}  rating {ex['rating']}  "
              f"{ex['solution_uci']}")


def fewshot_messages(examples, opts, mode):
    """The examples as earlier turns of the same chat: question, answer."""
    msgs = []
    for ex in examples:
        g = Game(ex)
        # the bare question: the system text is added once, by the caller
        plain = SimpleNamespace(**{**vars(opts), "system_in_user": False})
        user = build_prompt(g, plain, mode)[-1]
        msgs += [user, {"role": "assistant", "content": example_answer(ex, mode)}]
    return msgs


THINK_END = re.compile(r"</think>|</thought>", re.I)
MOVE_RE = re.compile(r"MOVE\s*:\s*[*`\"']*\s*([A-Za-z0-9=+#\-]+)", re.I)
LINE_RE = re.compile(r"LINE\s*:\s*[*`\"']*\s*([a-h1-8qrbnQRBN \t]+)", re.I)


def strip_thinking(text, truncated):
    """
    INPUT: the raw answer, whether generation hit the token limit
    OUTPUT: (the final answer without the reasoning, failure reason or "")

    A reasoning model writes <think> ... </think> before answering. Only what
    comes after the last </think> is the answer: a move written while
    "thinking aloud" is not a decision. An open <think> with no end means the
    model ran out of tokens while still reasoning.
    """
    parts = THINK_END.split(text)
    if len(parts) > 1:
        return parts[-1], ""
    if "<think>" in text.lower() or "<thought>" in text.lower():
        return "", "troncato mentre ragionava"
    if truncated and not MOVE_RE.search(text):
        return "", "troncato"
    return text, ""


def parse_move(text, board):
    """
    INPUT: the final answer, the current board
    OUTPUT: (chess.Move or None, failure reason)

    Takes the LAST "MOVE:" line (models sometimes restate a draft first), and
    accepts both UCI and SAN. Nothing else is guessed: an answer without a
    readable MOVE line is a failure, the same for every model.
    """
    found = MOVE_RE.findall(text)
    if not found:
        return None, "formato non rispettato"
    tok = found[-1].strip().rstrip(".")
    try:
        m = chess.Move.from_uci(tok.lower())
        if m in board.legal_moves:
            return m, ""
    except ValueError:
        pass
    try:
        return board.parse_san(tok), ""
    except ValueError:
        return None, "illegale"


def parse_line(text):
    """The player's own moves from a LINE: answer (every other token)."""
    found = LINE_RE.findall(text)
    if not found:
        return None
    toks = found[-1].split()
    out = []
    for t in toks:
        try:
            out.append(chess.Move.from_uci(t.lower()))
        except ValueError:
            return None
    return out[0::2]


class APIUnavailable(RuntimeError):
    """The server could not answer after every retry: the run must stop."""


class OpenAIBackend:
    """
    Any server speaking the OpenAI chat API:

      vLLM on Colab   --base-url http://localhost:8000/v1
      Gemini (free)   --base-url https://generativelanguage.googleapis.com/v1beta/openai
                      --api-key-env GEMINI_API_KEY --rpm 8

    Requests are sent in parallel; a local server batches them on the GPU.

    A network or quota failure is NOT a wrong answer. After every retry has
    failed the run stops with APIUnavailable and nothing is written for the
    puzzles still in play: they are replayed when the command is relaunched.
    Recording them as failures would charge the model for an outage.
    """

    def __init__(self, base_url, model, max_tokens, temperature, concurrency,
                 thinking, timeout, rpm=0, api_key_env="OPENAI_API_KEY",
                 top_p=None, top_k=None, reasoning_effort=None, retries=8,
                 think_budget=0):
        import requests
        self.requests = requests
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.max_tokens = model, max_tokens
        self.temperature, self.concurrency = temperature, concurrency
        self.thinking, self.timeout = thinking, timeout
        self.top_p, self.top_k = top_p, top_k
        self.reasoning_effort = reasoning_effort
        self.retries = retries
        self.think_budget = think_budget
        self.key = os.environ.get(api_key_env, "EMPTY")
        if self.key == "EMPTY" and "localhost" not in base_url \
                and "127.0.0.1" not in base_url:
            sys.exit(f"manca la chiave: imposta la variabile {api_key_env}")
        # rate limit: one request every 60/rpm seconds, shared by all threads
        self.interval = 60.0 / rpm if rpm else 0.0
        self.next_slot = 0.0
        self.lock = threading.Lock()
        self.abort = None        # set by the first fatal error: stops the rest

    def _wait_slot(self):
        if not self.interval:
            return
        with self.lock:
            now = time.time()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
        time.sleep(max(0.0, slot - now))

    def _body(self, messages):
        body = {"model": self.model, "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature}
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.top_k is not None:
            body["top_k"] = self.top_k                  # vLLM extension
        if self.thinking is not None:
            # vLLM, Qwen-style chat templates
            body["chat_template_kwargs"] = {"enable_thinking": self.thinking}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort  # Gemini, OpenAI
        return body

    def _post(self, body):
        """One request with retries; returns the JSON reply."""
        last = ""
        for attempt in range(self.retries):
            if self.abort:
                raise APIUnavailable(self.abort)
            self._wait_slot()
            if self.abort:
                raise APIUnavailable(self.abort)
            try:
                r = self.requests.post(
                    self.url, json=body, timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self.key}"})
            except Exception as e:                       # noqa: BLE001
                last = f"rete: {e}"
                time.sleep(min(120, 5 * 2 ** attempt))
                continue
            if r.status_code == 429 and re.search(r"per\s*day", r.text, re.I):
                # a DAILY quota does not come back by waiting minutes: stop
                # now instead of spending 40 minutes on hopeless retries
                m = re.search(r"Quota exceeded[^\\\n\"]*", r.text)
                self.abort = ("quota GIORNALIERA esaurita "
                              f"({m.group(0) if m else 'per day'}). Riprova "
                              "quando si azzera, o usa un altro modello")
                raise APIUnavailable(self.abort)
            if r.status_code == 429 or r.status_code >= 500:
                # quota or server busy: wait as long as the server asks
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                wait = r.headers.get("Retry-After")
                try:
                    wait = float(wait)
                except (TypeError, ValueError):
                    wait = min(300, 10 * 2 ** attempt)
                # the server's own words say WHICH limit was hit (per minute,
                # per day, overloaded...): print them, compacted to one line
                why = " ".join(r.text.split())
                # Google's message starts with boilerplate: show the part that
                # names the limit, when there is one
                k = why.find("Quota exceeded")
                why = why[k:k + 300] if k >= 0 else why[:300]
                print(f"    [{r.status_code}] attendo {wait:.0f} s "
                      f"(tentativo {attempt + 1}/{self.retries}): {why}",
                      flush=True)
                time.sleep(wait)
                continue
            if r.status_code >= 400:
                # a client error does not go away by retrying (wrong model
                # name, bad key, unsupported parameter): stop at once
                self.abort = f"HTTP {r.status_code}: {r.text[:500]}"
                raise APIUnavailable(self.abort)
            return r.json()
        raise APIUnavailable(f"nessuna risposta dopo {self.retries} "
                             f"tentativi. Ultimo errore: {last}")

    @staticmethod
    def _split(j):
        """(answer, reasoning, finish_reason, tokens) from a reply. The
        reasoning comes either apart (vLLM with --reasoning-parser) or inside
        the text between <think> and </think>."""
        ch = j["choices"][0]
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        # Gemma through the Gemini API writes its reasoning inline as
        # <thought>...</thought>: same thing as Qwen's <think>, other name
        text = text.replace("<thought>", "<think>").replace("</thought>",
                                                             "</think>")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if not reasoning and "<think>" in text:
            head, _, rest = text.partition("<think>")
            reasoning, closed, text = rest.partition("</think>")
            if not closed:                    # still thinking: no answer yet
                text = ""
        # generated tokens = total - prompt: Google leaves the thinking out of
        # completion_tokens (Gemma said "Hi!" with 2 completion tokens and 29
        # total), so completion_tokens alone would hide the real cost
        u = j.get("usage") or {}
        tok = (u.get("total_tokens", 0) - u.get("prompt_tokens", 0)
               if u.get("total_tokens") else u.get("completion_tokens", 0)) or 0
        return text, reasoning, ch.get("finish_reason"), tok

    FORCE = ("\n\nTime is up. I must stop calculating and answer now with "
             "the most promising move I have found.\n</think>\n\nMOVE: ")

    def _one(self, messages):
        """
        Without a thinking budget: one request; if the model runs out of
        tokens while still thinking, the answer is recorded as truncated.

        With --think-budget B (local servers only): the model thinks for at
        most B tokens. If it has not finished by then, its thinking is closed
        for it - "time is up, answer now" - and a second, tiny request asks
        only for the move (continue_final_message: the model continues our
        half-written answer, which already ends with "MOVE: "). Qwen describes
        this way of capping the thinking. Every question then costs at most
        B tokens and always gets an answer, like a player whose clock is
        running out.
        """
        body = self._body(messages)
        if self.think_budget:
            body["max_tokens"] = self.think_budget
        text, reasoning, finish, tok = self._split(self._post(body))
        unfinished = finish == "length" and not text.strip()
        if not (self.think_budget and unfinished and reasoning):
            if reasoning:
                text = ("<think>" + reasoning
                        + ("" if unfinished else "</think>") + text)
            return text, finish == "length", tok
        prefix = "<think>\n" + reasoning.strip() + self.FORCE
        body2 = dict(body, max_tokens=12, continue_final_message=True,
                     add_generation_prompt=False,
                     messages=messages + [{"role": "assistant",
                                           "content": prefix}])
        text2, reasoning2, _, tok2 = self._split(self._post(body2))
        forced = (text2 or reasoning2).strip()
        return ("<think>" + reasoning + "</think>\n[budget di ragionamento "
                "esaurito: risposta forzata]\nMOVE: " + forced,
                False, tok + tok2)

    def generate(self, batch):
        # a round can last many minutes with a slow API: print one line every
        # few answers so it is clear the run is alive and how fast it goes
        done = [0]
        t0 = time.time()
        step = max(1, len(batch) // 6)

        def one(messages):
            out = self._one(messages)
            with self.lock:
                done[0] += 1
                if done[0] % step == 0 or done[0] == len(batch):
                    print(f"    risposte {done[0]}/{len(batch)}  "
                          f"({time.time() - t0:.0f} s)", flush=True)
            return out

        with ThreadPoolExecutor(self.concurrency) as ex:
            return list(ex.map(one, batch))


class HFBackend:
    """
    transformers in this process, optionally in 4 bit (bitsandbytes). It works
    everywhere, but generates one batch at a time with no continuous
    batching: use it only if vLLM does not install.
    """

    def __init__(self, model, max_tokens, temperature, batch_size, four_bit,
                 thinking):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        half = torch.cuda.is_available()          # fp16 on GPU, fp32 on CPU
        kw = {"device_map": "auto",
              "torch_dtype": torch.float16 if half else torch.float32}
        if four_bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
        self.model.eval()
        self.max_tokens, self.temperature = max_tokens, temperature
        self.batch_size, self.thinking = batch_size, thinking

    def generate(self, batch):
        out = []
        extra = {} if self.thinking is None else {"enable_thinking": self.thinking}
        for i in range(0, len(batch), self.batch_size):
            prompts = [self.tok.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, **extra)
                for m in batch[i:i + self.batch_size]]
            # the chat template already contains BOS and friends: do not add
            # them twice. Only ids and mask go to generate(): some tokenizers
            # also return token_type_ids, which causal models reject.
            enc = self.tok(prompts, return_tensors="pt", padding=True,
                           add_special_tokens=False)
            enc = {"input_ids": enc["input_ids"].to(self.model.device),
                   "attention_mask": enc["attention_mask"].to(self.model.device)}
            sample = self.temperature > 0
            with self.torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=self.max_tokens, do_sample=sample,
                    temperature=self.temperature if sample else None,
                    top_p=0.95 if sample else None,
                    pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for row in new:
                n = int((row != self.tok.pad_token_id).sum())
                text = self.tok.decode(row, skip_special_tokens=True)
                out.append((text, n >= self.max_tokens, n))
        return out


class MockBackend:
    """No model: a random legal move, wrapped in fake reasoning, and now and
    then a broken answer. Exists only to test this script end to end."""

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def generate(self, batch):
        out = []
        for messages in batch:
            user = messages[-1]["content"]
            fen = re.search(r"Position \(FEN\): (.+)", user).group(1)
            board = chess.Board(fen)
            legal = sorted(board.legal_moves, key=lambda m: m.uci())
            m = self.rng.choice(legal)
            roll = self.rng.random()
            if "LINE:" in user:
                text = f"<think>hmm</think>\nLINE: {m.uci()}"
            elif roll < 0.05:
                text = "<think>still thinking about Qh5 and"      # truncated
            elif roll < 0.10:
                text = "I would play the queen somewhere."        # no format
            elif roll < 0.55:
                text = f"<think>MOVE: a1a1 no...</think>\nMOVE: {board.san(m)}"
            else:
                text = f"Analysis...\nMOVE: **{m.uci()}**"
            out.append((text, False, len(text) // 4))
        return out


class LLMPlayer:
    """
    mode "move": one call per move, the model sees the new position each time.
                 Same task as the GNN: this is the main comparison.
    mode "line": one call per puzzle, the model writes its whole line up front
                 and its moves are played in order whatever the defence does.
                 Cheaper; tests calculation without feedback.
    """

    def __init__(self, backend, opts, mode, keep_chars, examples=()):
        self.backend, self.opts, self.mode = backend, opts, mode
        self.keep = keep_chars
        # the same examples, in the same order, before every question: a
        # local server caches this common prefix, so it is almost free
        self.shots = fewshot_messages(examples, opts, mode)

    def choose(self, games):
        ask = [g for g in games if not (self.mode == "line" and g.plan is not None)]
        answers = self.backend.generate(
            [build_prompt(g, self.opts, self.mode, self.shots)
             for g in ask]) if ask else []
        for g, (text, truncated, ntok) in zip(ask, answers):
            g.calls += 1
            g.tokens += ntok
            g.responses.append(text if self.keep <= 0 else text[-self.keep:])
            final, why = strip_thinking(text, truncated)
            if self.mode == "line":
                g.plan = (parse_line(final) or []) if not why else []
                g._why = why or ("" if g.plan else "formato non rispettato")
            else:
                g._answer = (parse_move(final, g.board) if not why
                             else (None, why))
        out = []
        for g in games:
            if self.mode == "line":
                if g.k < len(g.plan):
                    out.append((g.plan[g.k], ""))
                else:
                    out.append((None, g._why or "linea finita"))
            else:
                out.append(g._answer)
        return out

    def close(self):
        pass


# ==========================================================================
# report
# ==========================================================================

def mcnemar(a, b):
    """Same formula as evaluate_models.mcnemar (continuity corrected)."""
    bb = sum(1 for x, y in zip(a, b) if x and not y)
    cc = sum(1 for x, y in zip(a, b) if y and not x)
    n = bb + cc
    if n == 0:
        return {"b": bb, "c": cc, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(bb - cc) - 1) ** 2 / n
    return {"b": bb, "c": cc, "chi2": chi2, "p": math.erfc(math.sqrt(chi2 / 2))}


def ci95(k, n):
    if not n:
        return 0.0
    p = k / n
    return 1.96 * math.sqrt(p * (1 - p) / n)


def band(n):
    """n >= 8 are pooled: there are only 25 such problems in all of Lichess."""
    return "8+" if n >= 8 else str(n)


def read_jsonl(path):
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r          # a re-run overwrites the old line
    return rows


def report(paths, out_path):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"file non trovati: {', '.join(missing)}. Quel giocatore "
                 f"non ha ancora giocato (oppure il percorso e' sbagliato)")
    runs = {}
    for p in paths:
        rows = read_jsonl(p)
        name = next(iter(rows.values()))["player"] if rows else p
        runs[name] = rows
    names = list(runs)
    empty = [n for n, r in runs.items() if not r]
    if empty:
        sys.exit(f"file senza risultati: {', '.join(empty)}. "
                 f"Fai prima giocare quel giocatore (play), poi rifai il report")
    common = set.intersection(*(set(r) for r in runs.values()))
    if not common:
        sys.exit("nessun problema in comune fra i file: non c'e' niente da "
                 "confrontare")
    print(f"giocatori: {', '.join(names)}")
    print(f"problemi in comune a tutti: {len(common)}"
          + ("" if all(len(r) == len(common) for r in runs.values())
             else "   (attenzione: non tutti hanno finito)"))

    rep = {"players": names, "common": len(common), "by": {}}
    for key, label in (("n", "per n"), ("source", "per fonte")):
        print(f"\n{'=' * 78}\nRISOLTI {label.upper()}  (in parentesi: prima mossa giusta)\n"
              f"{'=' * 78}")
        groups = defaultdict(list)
        for i in common:
            r = runs[names[0]][i]
            groups[band(r["n"]) if key == "n" else r["source"]].append(i)
        order = (sorted(groups, key=lambda s: (len(s), s)) if key == "n"
                 else sorted(groups))
        print(f"{'':>13} {'prob':>5} | " + " | ".join(f"{n[:18]:>18}" for n in names))
        rep["by"][key] = {}
        for gname in order + ["tot"]:
            ids = list(common) if gname == "tot" else groups[gname]
            cells, row = [], {"problems": len(ids)}
            for n in names:
                k = sum(runs[n][i]["solved"] for i in ids)
                f = sum(runs[n][i]["first_ok"] for i in ids)
                row[n] = {"solved": k, "acc": k / len(ids), "ci": ci95(k, len(ids)),
                          "first_ok": f / len(ids)}
                cells.append(f"{k / len(ids):>7.1%} ({f / len(ids):>6.1%})  ")
            rep["by"][key][gname] = row
            print(f"{gname:>13} {len(ids):>5} | " + " | ".join(f"{c:>18}" for c in cells))

    # n and source crossed: for n = 2..4 the "per n" rows mix classic and
    # Lichess problems, so a drop there could come from either. Only the
    # crossed table says which.
    print(f"\n{'=' * 78}\nPER N E FONTE  (risolti / prima mossa giusta, "
          f"[quanti problemi])\n{'=' * 78}")
    first = runs[names[0]]
    sources = sorted({first[i]["source"] for i in common})
    bands = sorted({band(first[i]["n"]) for i in common},
                   key=lambda s: (len(s), s))
    rep["by"]["n_source"] = {}
    for n in names:
        print(f"\n{n}")
        print(f"{'n':>4} | " + " | ".join(f"{s:>22}" for s in sources))
        rep["by"]["n_source"][n] = {}
        for b in bands:
            cells = []
            for s in sources:
                ids = [i for i in common if band(first[i]["n"]) == b
                       and first[i]["source"] == s]
                if not ids:
                    cells.append(f"{'-':>22}")
                    continue
                k = sum(runs[n][i]["solved"] for i in ids) / len(ids)
                f = sum(runs[n][i]["first_ok"] for i in ids) / len(ids)
                rep["by"]["n_source"][n][f"{b}|{s}"] = {
                    "problems": len(ids), "acc": k, "first_ok": f}
                cells.append(f"{k:>6.1%} / {f:>6.1%} [{len(ids):>2}]")
            print(f"{b:>4} | " + " | ".join(f"{c:>22}" for c in cells))

    print(f"\n{'=' * 78}\nPERCHE' FALLISCONO\n{'=' * 78}")
    rep["reasons"] = {}
    for n in names:
        c = Counter(runs[n][i]["reason"] for i in common if not runs[n][i]["solved"])
        rep["reasons"][n] = dict(c)
        print(f"{n}: " + (", ".join(f"{k} {v}" for k, v in c.most_common()) or "-"))

    print(f"\n{'=' * 78}\nCOSTO MEDIO PER PROBLEMA\n{'=' * 78}")
    rep["cost"] = {}
    for n in names:
        rs = [runs[n][i] for i in common]
        s = sum(r["seconds"] for r in rs) / len(rs)
        t = sum(r["tokens"] for r in rs) / len(rs)
        rep["cost"][n] = {"seconds": s, "tokens": t}
        print(f"{n:>20}: {s:8.1f} s   {t:8.0f} token")

    if len(names) > 1:
        print(f"\n{'=' * 78}\nMcNEMAR, problema per problema (b = vince il primo, "
              f"c = vince il secondo)\n{'=' * 78}")
        ids = sorted(common)
        rep["mcnemar"] = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                mc = mcnemar([runs[a][x]["solved"] for x in ids],
                             [runs[b][x]["solved"] for x in ids])
                rep["mcnemar"][f"{a} vs {b}"] = mc
                star = "  *" if mc["p"] < 0.05 else ""
                print(f"{a:>20} vs {b:<20} b={mc['b']:<4} c={mc['c']:<4} "
                      f"p={mc['p']:.2g}{star}")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(rep, f, indent=1)
        print(f"\nreport salvato in {out_path}")


def recheck(paths, heldout, sf_path, nodes, threads):
    """Re-search every "matto perso" verdict with a far larger node budget.

    The judge uses 2M nodes to keep a run fast; a deep mate it cannot prove in
    that budget would be a FALSE "matto perso" costing the player a deserved
    point. This replays each such game up to the verdict and searches again with
    a much larger budget (and more threads: here only the answer matters, not
    reproducibility). If nothing is overturned, the judge's budget was enough.
    """
    df = pd.read_csv(heldout).set_index("id")
    df.index = df.index.astype(str)
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"Threads": threads, "Hash": 512})
    try:
        for p in paths:
            rows = [r for r in read_jsonl(p).values()
                    if r["reason"] == "matto perso"]
            wrong = []
            print(f"\n{p}: {len(rows)} verdetti 'matto perso' da ricontrollare")
            for i, r in enumerate(rows, 1):
                board = chess.Board(df.loc[r["id"], "fen"])
                solver = board.turn
                for u in r["moves"].split():
                    board.push_uci(u)
                remaining = r["n"] - r["moves_made"]
                info = engine.analyse(board, chess.engine.Limit(
                    mate=remaining, nodes=nodes))
                mate = info["score"].pov(solver).mate()
                ok = mate is not None and 0 < mate <= remaining
                if ok:
                    wrong.append(r["id"])
                print(f"  {i:>4}/{len(rows)} {r['id']:<16} n={r['n']:<2} "
                      f"dopo {r['moves_made']} mosse, ne restano {remaining}: "
                      f"{'GIUDICE SMENTITO, matto in ' + str(mate) if ok else 'confermato'}",
                      flush=True)
            print(f"  => verdetti sbagliati: {len(wrong)} su {len(rows)}"
                  + (f": {', '.join(wrong)}" if wrong else ""))
    finally:
        engine.quit()


# ==========================================================================
# main
# ==========================================================================

def make_player(a):
    if a.player == "stockfish":
        return StockfishPlayer(a.stockfish, a.player_nodes or a.judge_nodes)
    if a.player == "random":
        return RandomPlayer(a.seed)
    if a.player == "gnn":
        if not a.ckpt:
            sys.exit("--player gnn vuole --ckpt")
        return GNNPlayer(a.ckpt, a.code_dir, a.batch_size)
    if a.backend == "openai":
        if not a.model:
            sys.exit("--backend openai vuole --model (il nome servito dal server)")
        be = OpenAIBackend(a.base_url, a.model, a.max_tokens, a.temperature,
                           a.concurrency, a.thinking, a.timeout, a.rpm,
                           a.api_key_env, a.top_p, a.top_k,
                           a.reasoning_effort, retries=a.retries,
                           think_budget=a.think_budget)
    elif a.backend == "hf":
        if not a.model:
            sys.exit("--backend hf vuole --model (es. Qwen/Qwen3-8B)")
        be = HFBackend(a.model, a.max_tokens, a.temperature, a.batch_size,
                       a.four_bit, a.thinking)
    else:
        be = MockBackend(a.seed)
    opts = SimpleNamespace(board=not a.no_board, legal=not a.no_legal,
                           system_in_user=a.system_in_user)
    examples = load_examples(a.shots_file, a.heldout)
    print(f"esempi nel prompt (few-shot): {len(examples)}")
    return LLMPlayer(be, opts, a.mode, a.keep_chars, examples)


def default_name(a):
    if a.player == "gnn":
        return "gnn_" + os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    if a.player == "llm":
        model = (a.model or "mock").split("/")[-1]
        return f"llm_{model}_{a.mode}"
    return a.player


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="fa giocare un giocatore su tutto il set")
    p.add_argument("--player", required=True,
                   choices=["stockfish", "random", "gnn", "llm"])
    p.add_argument("--heldout", required=True)
    p.add_argument("--out", required=True, help="file JSONL dei risultati")
    p.add_argument("--name", help="nome del giocatore nel report")
    p.add_argument("--limit", type=int)
    p.add_argument("--n-max", type=int)
    p.add_argument("--ids", nargs="+", help="solo questi problemi")
    p.add_argument("--seed", type=int, default=0)
    # referee
    p.add_argument("--stockfish", default="stockfish")
    p.add_argument("--defence", choices=["stockfish", "reference"],
                   default="stockfish")
    p.add_argument("--defence-nodes", type=int, default=300_000)
    p.add_argument("--judge-nodes", type=int, default=2_000_000)
    p.add_argument("--escalate-nodes", type=int, default=20_000_000,
                   help="seconda ricerca, piu' lunga, se la posizione sembra "
                        "ancora vinta (0 = mai)")
    p.add_argument("--workers", type=int,
                   default=max(1, min(8, (os.cpu_count() or 2) - 1)),
                   help="processi Stockfish in parallelo per l'arbitro")
    p.add_argument("--player-nodes", type=int,
                   help="solo per --player stockfish (default: come il giudice)")
    # gnn
    p.add_argument("--ckpt")
    p.add_argument("--code-dir", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--batch-size", type=int, default=64)
    # llm
    p.add_argument("--backend", choices=["openai", "hf", "mock"], default="mock")
    p.add_argument("--model")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--mode", choices=["move", "line"], default="move")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--thinking", choices=["on", "off", "default"], default="default")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--four-bit", action="store_true")
    p.add_argument("--no-board", action="store_true")
    p.add_argument("--no-legal", action="store_true")
    p.add_argument("--retries", type=int, default=8,
                   help="tentativi per ogni richiesta su errori 429/5xx")
    p.add_argument("--rpm", type=int, default=0,
                   help="richieste al minuto al massimo (0 = nessun limite)")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY",
                   help="variabile d'ambiente con la chiave dell'API")
    p.add_argument("--top-p", type=float)
    p.add_argument("--top-k", type=int)
    p.add_argument("--reasoning-effort",
                   choices=["none", "low", "medium", "high"],
                   help="per le API che lo supportano (Gemini, OpenAI)")
    p.add_argument("--per-n", type=int,
                   help="pilota: solo K problemi per ogni fascia di n, "
                        "scelti a caso col seed")
    p.add_argument("--think-budget", type=int, default=0,
                   help="tetto ai token di ragionamento; poi risposta forzata "
                        "(solo server locali come vLLM; 0 = nessun tetto)")
    p.add_argument("--system-in-user", action="store_true",
                   help="per i modelli senza ruolo system (es. Gemma via API): "
                        "le istruzioni vanno in testa al primo messaggio")
    p.add_argument("--shots-file",
                   help="esempi risolti da mettere nel prompt (few-shot), "
                        "creati con il comando fewshot")
    p.add_argument("--keep-chars", type=int, default=0,
                   help="quanti caratteri tenere di ogni risposta (0 = tutti)")

    r = sub.add_parser("report", help="tabelle e McNemar fra giocatori")
    r.add_argument("files", nargs="+")
    r.add_argument("--out")

    f = sub.add_parser("fewshot", help="sceglie dal training gli esempi "
                                       "risolti per il prompt")
    f.add_argument("--train-csv", required=True)
    f.add_argument("--heldout", required=True)
    f.add_argument("--n", type=int, nargs="+", default=[1, 2, 3])
    f.add_argument("--rating-min", type=int, default=1200)
    f.add_argument("--rating-max", type=int, default=1800)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out", default="fewshot.json")

    q = sub.add_parser("prompt", help="mostra il prompt esatto di un problema")
    q.add_argument("--heldout", required=True)
    q.add_argument("--id", required=True)
    q.add_argument("--shots-file")
    q.add_argument("--mode", choices=["move", "line"], default="move")

    c = sub.add_parser("recheck", help="ricontrolla i 'matto perso' con un "
                                       "budget molto piu' grande")
    c.add_argument("files", nargs="+")
    c.add_argument("--heldout", required=True)
    c.add_argument("--stockfish", default="stockfish")
    c.add_argument("--nodes", type=int, default=50_000_000)
    c.add_argument("--threads", type=int, default=4)

    a = ap.parse_args()
    if a.cmd == "fewshot":
        make_fewshot(a.train_csv, a.heldout, a.n, a.rating_min, a.rating_max,
                     a.seed, a.out)
        return
    if a.cmd == "prompt":
        games = load_games(a.heldout, ids=[a.id])
        if not games:
            sys.exit(f"{a.id} non e' nel held-out")
        opts = SimpleNamespace(board=True, legal=True)
        shots = fewshot_messages(load_examples(a.shots_file, a.heldout),
                                 opts, a.mode)
        for m in build_prompt(games[0], opts, a.mode, shots):
            print(f"\n----- {m['role'].upper()} -----\n{m['content']}")
        return
    if a.cmd == "report":
        report(a.files, a.out)
        return
    if a.cmd == "recheck":
        recheck(a.files, a.heldout, a.stockfish, a.nodes, a.threads)
        return

    a.thinking = {"on": True, "off": False, "default": None}[a.thinking]
    name = a.name or default_name(a)

    games = load_games(a.heldout, a.limit, a.n_max, a.ids, a.per_n, a.seed)
    done = set(read_jsonl(a.out)) if os.path.exists(a.out) else set()
    todo = [g for g in games if g.id not in done]
    print(f"giocatore: {name}")
    print(f"problemi: {len(games)}, gia' fatti: {len(games) - len(todo)}, "
          f"da fare: {len(todo)}")
    if not todo:
        print("niente da fare: usa `report` per i risultati")
        return
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # the player first: if it fails to load (missing module, wrong
    # checkpoint) no Stockfish process has been started yet, so nothing is
    # left hanging and the script exits cleanly
    player = make_player(a)
    ref = None
    t0 = time.time()
    try:
        ref = Referee(a.stockfish, a.defence_nodes, a.judge_nodes, a.defence,
                      a.escalate_nodes, a.workers)
        print(f"arbitro: {ref.workers} processi Stockfish")
        play_all(ref, player, todo, a.out, name)
    except APIUnavailable as e:
        sys.exit(f"\n>>> API non disponibile, run fermata: {e}\n"
                 f">>> i problemi gia' finiti sono salvati in {a.out}; "
                 f"rilancia lo stesso comando per riprendere")
    finally:
        player.close()
        if ref is not None:
            ref.close()

    rows = read_jsonl(a.out)
    k = sum(rows[g.id]["solved"] for g in games if g.id in rows)
    if ref is not None:
        print(f"\nricerche lunghe del giudice: {ref.escalations}")
    print(f"finito in {time.time() - t0:.0f} s: {k}/{len(games)} risolti "
          f"({k / len(games):.1%}). Risultati in {a.out}")


if __name__ == "__main__":
    main()