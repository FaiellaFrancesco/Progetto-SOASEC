"""
playout_models.py

La seconda metrica: il modello gioca la LINEA INTERA fino al matto, non solo la
prima mossa.

    python playout_models.py \\
        --timing-ckpt ../../ckpt/timing/best.pt \\
        --plain-ckpt  ../../ckpt/plain/best.pt \\
        --test-csv ../../dataset/test.csv --limit 20000

PERCHE' NON BASTA evaluate.py
------------------------------
evaluate.py fa il playout con un `move_fn(board) -> chess.Move`, e quel
contratto va benissimo per l'LLM o per la baseline. Ma il nostro modello, con
il timing acceso, ha bisogno di sapere DA QUANTE SEMI-MOSSE non si muove ogni
pezzo, e una board nuda quella storia non ce l'ha: la si perde appena si esce
dalla prima mossa.

Qui la storia viene mantenuta mano a mano che la partita avanza, con le stesse
identiche regole di row_to_graph: `played` conta le semi-mosse, `last_moved`
tiene l'ultimo ply in cui ogni casella e' stata occupata da un pezzo in
movimento. Se non lo facessimo, dalla seconda mossa in poi il modello con il
tempo riceverebbe valori sbagliati e lo valuteremmo in condizioni peggiori di
quelle in cui e' stato addestrato.

PERCHE' A LOTTI
---------------
Un playout e' sequenziale: la mossa 2 dipende dalla 1. Fatto un puzzle alla
volta significa un forward con batch=1, e su 193.000 puzzle sono ore.
Qui si portano avanti TUTTI i puzzle in parallelo: a ogni giro si costruisce un
batch con la posizione corrente di ognuno, un solo forward decide per tutti, e
chi ha finito esce. Stesso risultato, due ordini di grandezza piu' veloce.

SEVERITA'
---------
Un puzzle conta come risolto solo se il modello gioca TUTTE le mosse della
linea registrata e la posizione finale e' scacco matto. Alla prima deviazione
e' fallito, anche se la mossa alternativa desse comunque matto: senza un motore
che giochi la difesa non possiamo verificarlo. E' quindi un LIMITE INFERIORE
del vero tasso di risoluzione, e va dichiarato come tale.
"""

import argparse
import json
import math

import chess
import numpy as np
import torch
from torch_geometric.data import Batch

from evaluate_models import load_model, mcnemar
from fen_to_graph import (normalize_row, get_puzzle_position, decode_move,
                          build_node_features, build_edge, build_legal_moves,
                          simulate_think_time, build_edge_time,
                          update_last_moved)
from graph_dataset import piece_ids_from_x, N_CLASSES

import pandas as pd


"""
INPUT: la soluzione restituita da get_puzzle_position
OUTPUT: (mosse del solver, risposte della difesa)

Le pari sono le mosse da indovinare, le dispari le risposte registrate da
Lichess. Duplicata da evaluate.py di proposito: cosi' questo script gira anche
senza quel file, e le due copie sono due righe identiche che non divergeranno.
"""


def solver_and_defence(solution):
    return solution[0::2], solution[1::2]


class Puzzle:
    """Lo stato di un puzzle mentre viene giocato."""

    def __init__(self, board, solution, rating, n_total, setup_uci):
        # la posizione di partenza va conservata intatta: giocare un puzzle lo
        # modifica, e lo stesso puzzle viene giocato una volta per modello.
        # Senza questa copia il secondo modello partirebbe dalle posizioni in
        # cui il primo ha lasciato la partita.
        self.board0 = board.copy()
        self.solver_moves, self.defence = solver_and_defence(solution)
        self.rating = rating
        self.n_total = n_total
        self.setup_uci = setup_uci
        self.reset()

    def reset(self):
        """Riporta il puzzle alla posizione iniziale, storia compresa."""
        self.board = self.board0.copy()
        self.k = 0                 # mosse del solver gia' giocate
        self.done = False
        self.solved = False
        # la storia, inizializzata come in row_to_graph: la mossa di setup e'
        # gia' stata giocata da get_puzzle_position ed e' il ply 1
        self.played = 1
        self.last_moved = {}
        update_last_moved(self.last_moved,
                          chess.Move.from_uci(self.setup_uci), self.played)

    def push(self, uci):
        move = chess.Move.from_uci(uci)
        self.board.push(move)
        self.played += 1
        update_last_moved(self.last_moved, move, self.played)

    @property
    def n_remaining(self):
        return self.n_total - self.k


"""
INPUT: un Puzzle nel suo stato corrente, e il flag del timing
OUTPUT: un torch_geometric.data.Data della posizione attuale

Costruisce lo stesso grafo che l'encoder avrebbe prodotto per questa
posizione, storia inclusa. Se questa funzione e row_to_graph divergono, il
modello viene valutato su input diversi da quelli su cui e' stato addestrato.
"""


def puzzle_to_data(p, with_time):
    from torch_geometric.data import Data

    x = build_node_features(p.board)
    edge_index, edge_attr = build_edge(p.board)
    legal = build_legal_moves(p.board)

    if with_time:
        t = simulate_think_time(p.rating, p.n_remaining, len(legal))
        x = np.concatenate([x, np.full((64, 1), t, dtype=np.float32)], axis=1)
        times = build_edge_time(edge_index, p.last_moved, p.played)
    else:
        times = np.zeros(edge_index.shape[1], dtype=np.float32)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        event_ids=piece_ids_from_x(x),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_type=torch.tensor(np.argmax(edge_attr, axis=1), dtype=torch.long),
        time=torch.tensor(times, dtype=torch.float32),
        legal_moves=torch.tensor(np.asarray(legal), dtype=torch.long),
        n_legal=torch.tensor([len(legal)], dtype=torch.long),
    )


def batch_legal_mask(batch, device):
    b = batch.n_legal.shape[0]
    mask = torch.full((b, N_CLASSES), float("-inf"), device=device)
    off = 0
    for i, k in enumerate(batch.n_legal.tolist()):
        mask[i, batch.legal_moves[off:off + k]] = 0.0
        off += k
    return mask


"""
INPUT: il CSV di test, un limite di righe
OUTPUT: la lista dei Puzzle pronti da giocare

Scarta le righe che l'encoder scarterebbe, cosi' il playout misura lo stesso
insieme di puzzle su cui il modello e' stato addestrato e valutato.
"""


def load_puzzles(csv_path, limit=None):
    df = pd.read_csv(csv_path, nrows=limit)
    out = []
    for row in df.to_dict("records"):
        r = normalize_row(row)
        moves = r["Moves"]
        if len(moves) < 2:
            continue
        n_total = len(moves) // 2
        try:
            tagged = int(r["MateIn"])
        except (KeyError, ValueError, TypeError):
            tagged = None
        if tagged is not None and tagged > 0 and tagged != n_total:
            continue
        board, solution = get_puzzle_position(r)
        if board is None:
            continue
        if any(len(m) == 5 and m[4] in "rbn" for m in solution[::2]):
            continue
        replay = board.copy()
        try:
            for uci in solution:
                replay.push_uci(uci)
        except ValueError:
            continue
        out.append(Puzzle(board, solution, r["Rating"], n_total, moves[0]))
    return out


"""
INPUT: un modello, la lista dei puzzle, il flag del timing, il device
OUTPUT: array booleano: per ogni puzzle, se e' stato risolto per intero

Porta avanti tutti i puzzle in parallelo. A ogni giro un solo forward decide
la mossa per tutti quelli ancora vivi.
"""


def playout(model, puzzles, with_time, device, batch_size=256, verbose=True):
    # reset COMPLETO: board e storia, non solo i contatori. Il playout
    # precedente ha lasciato le partite a meta'.
    for p in puzzles:
        p.reset()

    active = [p for p in puzzles if p.solver_moves]
    giro = 0
    with torch.no_grad():
        while active:
            giro += 1
            if verbose:
                print(f"    mossa {giro}: {len(active):,} puzzle ancora in gioco",
                      flush=True)
            nxt = []
            for i in range(0, len(active), batch_size):
                chunk = active[i:i + batch_size]
                batch = Batch.from_data_list(
                    [puzzle_to_data(p, with_time) for p in chunk]).to(device)
                mask = batch_legal_mask(batch, device)
                pred = (model(batch) + mask).argmax(dim=1).cpu().tolist()

                for p, y in zip(chunk, pred):
                    move = decode_move(int(y), p.board)
                    expected = p.solver_moves[p.k]
                    if move.uci()[:4] != expected[:4]:
                        p.done = True          # deviazione: fallito
                        continue
                    p.push(expected)
                    if p.k < len(p.defence):
                        p.push(p.defence[p.k])
                    p.k += 1
                    if p.k >= len(p.solver_moves):
                        p.done = True
                        p.solved = p.board.is_checkmate()
                    else:
                        nxt.append(p)
            active = nxt
    return np.array([p.solved for p in puzzles], dtype=bool)


def summarise(puzzles, solved):
    depth = np.array([p.n_total for p in puzzles])
    out = {}
    for d in sorted(set(depth.tolist())):
        m = depth == d
        k, n = int(solved[m].sum()), int(m.sum())
        a = k / n if n else 0.0
        out[int(d)] = {"n": n, "solved": k, "acc": a,
                       "ci": 1.96 * math.sqrt(a * (1 - a) / n) if n else 0.0}
    k, n = int(solved.sum()), len(solved)
    a = k / n
    out["overall"] = {"n": n, "solved": k, "acc": a,
                      "ci": 1.96 * math.sqrt(a * (1 - a) / n)}
    return out, depth


def print_table(title, s):
    print(f"\n{title}")
    print(f"{'n':>3} {'puzzle':>9} {'risolti':>9} {'%':>10} {'+/- 95%':>9}")
    for d in sorted(k for k in s if k != "overall"):
        r = s[d]
        print(f"{d:>3} {r['n']:>9,} {r['solved']:>9,} {r['acc']:>9.2%} "
              f"{r['ci']:>8.2%}")
    o = s["overall"]
    print(f"{'tot':>3} {o['n']:>9,} {o['solved']:>9,} {o['acc']:>9.2%} "
          f"{o['ci']:>8.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing-ckpt", required=True)
    ap.add_argument("--plain-ckpt", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="quante righe del CSV usare (tutto se omesso)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="results_playout.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    puzzles = load_puzzles(args.test_csv, args.limit)
    print(f"puzzle validi: {len(puzzles):,}")

    res, results = {}, {}
    for name, ck, with_time in (("timing", args.timing_ckpt, True),
                                ("plain", args.plain_ckpt, False)):
        model, a, epoch, best_val = load_model(ck, device)
        assert a["timing"] == with_time, f"{name}: checkpoint sbagliato"
        print(f"\n{name}: epoca {epoch}, val loss {best_val:.4f}")
        solved = playout(model, puzzles, with_time, device, args.batch_size)
        res[name] = solved
        results[name], depth = summarise(puzzles, solved)
        print_table(f"PLAYOUT - {name}", results[name])

    print("\nCONFRONTO (McNemar)")
    print(f"{'n':>3} {'plain':>9} {'timing':>9} {'diff':>9} {'b':>6} {'c':>6} "
          f"{'p':>10}")
    results["mcnemar"] = {}
    for d in sorted(k for k in results["timing"] if k != "overall"):
        m = depth == d
        mc = mcnemar(res["timing"][m], res["plain"][m])
        results["mcnemar"][int(d)] = mc
        at, ap_ = results["timing"][d]["acc"], results["plain"][d]["acc"]
        star = "*" if mc["p"] < 0.05 else ""
        print(f"{d:>3} {ap_:>8.2%} {at:>9.2%} {at-ap_:>+9.2%} {mc['b']:>6,} "
              f"{mc['c']:>6,} {mc['p']:>10.2} {star}")
    mc = mcnemar(res["timing"], res["plain"])
    results["mcnemar"]["overall"] = mc
    at = results["timing"]["overall"]["acc"]
    ap_ = results["plain"]["overall"]["acc"]
    print(f"{'tot':>3} {ap_:>8.2%} {at:>9.2%} {at-ap_:>+9.2%} {mc['b']:>6,} "
          f"{mc['c']:>6,} {mc['p']:>10.2} {'*' if mc['p']<0.05 else ''}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"\nsalvato in {args.out}")
    print("\nNB: un puzzle conta risolto solo se il modello gioca tutta la linea")
    print("    registrata e la posizione finale e' matto. E' un limite inferiore.")


if __name__ == "__main__":
    main()
