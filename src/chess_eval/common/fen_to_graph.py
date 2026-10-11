# Library module (imported, not run directly).
# Used by: preprocessing.db_rows_to_graphs, preprocessing.db_dropped_stats,
#          evaluation.playout_models, tests.tests
#
# Encodes a lichess puzzle (a FEN position) as a typed graph for the GNN:
# 64 nodes, one per square, carrying piece / colour / position features, and
# directed edges of four kinds (attacks, defends, moves, pushes). The GNN gets
# the piece relationships explicitly as edges; the LLM only sees the FEN text.
#
# Example input row (lichess puzzle CSV):
#   000Zo,4r3/1k6/pp3r2/1b2P2p/3R1p2/P1R2P2/1P4PP/6K1 w - - 0 35,e5f6 e8e1 g1f2 e1f1,1363,76,86,655,endgame mate mateIn2 operaMate short,https://lichess.org/n8Ff742v#69,,

import chess
import numpy as np

NODE_FEATURES = ["pawn", "knight", "bishop", "rook", "queen",
                 "king", "white", "black", "empty", "file", "rank", "white_to_move"]
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK,
               chess.QUEEN, chess.KING]  # order of the piece-type one-hot columns
EDGE_TYPES = ["attacks", "defends", "moves", "pushes"]

UNKNOWN_RECENCY = 20  # recency for a piece that never moved or moved long ago


def normalize_row(row):
    """Split the space-joined Moves/Themes strings and make Rating an int."""
    r = dict(row)
    if isinstance(r["Moves"], str):
        r["Moves"] = r["Moves"].split()
    if isinstance(r["Themes"], str):
        r["Themes"] = r["Themes"].split()
    if isinstance(r["Rating"], str):
        r["Rating"] = int(r["Rating"])
    return r


def get_puzzle_position(row):
    """Build the board and play the setup move, returning (board, remaining moves).

    The FEN sits one move before the puzzle starts, so the first move in the
    row is the opponent's setup: we play it and hand back the solver's line.
    Returns (None, None) if the FEN is invalid.
    """
    try:
        board = chess.Board(row["FEN"])
        board.push_uci(row["Moves"][0])
        return board, row["Moves"][1:]
    except ValueError:
        print(f"Invalid FEN: {row['FEN']}")
        return None, None


def pawn_pushes(board, square, piece):
    """Squares the pawn on `square` can advance to (one, or two from the start).

    board.legal_moves only covers the side to move; here we need pushes for
    both colours, so we compute them by hand.
    """
    if piece.color == chess.WHITE:
        step = 8
        start_rank = 1
    else:
        step = -8
        start_rank = 6
    targets = []
    one = square+step
    if 0 <= one < 64 and board.piece_at(one) is None:
        targets.append(one)
        two = square+2*step
        if chess.square_rank(square) == start_rank and 0 <= two < 64 and board.piece_at(two) is None:
            targets.append(two)

    return targets


def build_node_features(board):
    """Node features, shape (64, 12): one row per square.

    Each square is a node (a1=0, b1=1, ..., h8=63) described by 12 numbers:
    piece type, colour, empty flag, file, rank, and whose turn it is.
    """
    x = np.zeros((64, len(NODE_FEATURES)), dtype=np.float32)

    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            x[square, 8] = 1.0                              # empty
        else:
            x[square, PIECE_TYPES.index(piece.piece_type)] = 1.0
            if piece.color == chess.WHITE:
                x[square, 6] = 1.0                          # white piece
            else:
                x[square, 7] = 1.0                          # black piece

        x[square, 9] = square % 8 / 7.0                     # file, in [0,1]
        x[square, 10] = square // 8 / 7.0                   # rank, in [0,1]
        if board.turn == chess.WHITE:
            x[square, 11] = 1.0                             # white to move
        else:
            x[square, 11] = 0.0                             # black to move
    return x


def build_edge(board):
    """Directed edges of the position.

    Returns edge_index (2, E) and a one-hot edge_attr (E, 4). An edge runs from
    a piece's square to each square it attacks: type defends (own piece),
    attacks (enemy) or moves (empty). Pawns also get a pushes edge.
    """
    sources, targets, kinds = [], [], []

    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            continue

        for target in board.attacks(square):
            occupant = board.piece_at(target)
            if occupant is None:
                kind = 2
            elif occupant.color == piece.color:
                kind = 1
            else:
                kind = 0
            sources.append(square)
            targets.append(target)
            kinds.append(kind)

        if piece.piece_type == chess.PAWN:
            for target in pawn_pushes(board, square, piece):
                sources.append(square)
                targets.append(target)
                kinds.append(3)

    edge_index = np.array([sources, targets], dtype=np.int64)

    edge_attr = np.zeros((len(kinds), len(EDGE_TYPES)), dtype=np.float32)
    edge_attr[np.arange(len(kinds)), kinds] = 1.0

    return edge_index, edge_attr


def build_label(solution):
    """Encode the first move as a class 0..4095: from_square * 64 + to_square.

    E.g. "e8e1" is from 60 to 4 -> 3844, since 3844 // 64 = 60 and 3844 % 64 = 4.
    """
    move = chess.Move.from_uci(solution[0])
    return move.from_square * 64 + move.to_square


def decode_move(y, board):
    """Inverse of build_label: class 0..4095 -> a chess.Move ready to push.

    build_label and decode_move are a pair and must agree. With 4096 classes
    (one per from/to pair) the label cannot say which piece a pawn promotes to,
    so we fix the same convention the encoder uses: a pawn reaching rank 1 or 8
    always becomes a queen. Under-promotion puzzles are dropped in row_to_graph,
    so the two never disagree. The board is needed only to tell whether the
    piece on the source square is a pawn.
    """
    src = y // 64
    dst = y % 64
    piece = board.piece_at(src)
    if (piece is not None and piece.piece_type == chess.PAWN
            and chess.square_rank(dst) in (0, 7)):
        return chess.Move(src, dst, promotion=chess.QUEEN)
    return chess.Move(src, dst)


def build_legal_moves(board):
    """Legal moves of the position, encoded like y (from*64 + to), deduplicated.

    The model does not know the rules of chess, so this is used to mask it.
    """
    idx = [m.from_square * 64 + m.to_square for m in board.legal_moves]
    return np.unique(np.array(idx, dtype=np.int16))


def simulate_think_time(rating, n_remaining, n_legal):
    """A plausible human think time for the position, normalised to [0, 1].

    The constants are hand-tuned for believable timing: slower on harder
    puzzles, slower on the first move, slower with more options.
    """
    base = 2.0 + 18.0 * (rating - 400) / 2600.0          # ~2..20 s over the rating range
    depth = 1.0 + 0.35 * (n_remaining - 1)               # the first move takes longest
    branch = 1.0 + 0.02 * (n_legal - 20)                 # more options, more time
    seconds = base * depth * branch
    return min(max(seconds / 60.0, 0.0), 1.0)


def row_to_graph(row, use_timing=False, unroll=True):
    """Turn one puzzle row into a list of training examples (empty if unusable).

    A mate-in-n puzzle is 2n moves (solver n, opponent n-1, plus the setup
    move at index 0). With unroll=True we emit one example per solver decision
    (mate-in-n, then n-1, ...), used for train/val; with unroll=False only the
    first move, used for the test set. The label y is always a single move;
    the full line is kept as metadata under "solution".

    Lichess only tags mateIn1..5, so rows tagged with a different depth than
    their move count are skipped, as are under-promotion solutions.
    """
    row = normalize_row(row)
    moves = row["Moves"]
    n_total = len(moves) // 2                             # solver moves to mate
    if n_total < 1:
        return []
    try:
        tagged = int(row["MateIn"])
    except (KeyError, ValueError, TypeError):
        tagged = None
    if tagged is not None and tagged > 0 and tagged != n_total:
        return []                                         # tag disagrees with move count

    board, solution = get_puzzle_position(row)
    if board is None:
        return []

    if any(len(m) == 5 and m[4] in "rbn" for m in solution[::2]):
        return []                                         # keep only queen promotions

    played = 1
    last_moved = {}
    update_last_moved(last_moved, chess.Move.from_uci(moves[0]), played)

    solver = board.turn                                   # the side to move is the solver

    examples = []
    k = 0                                                 # solver moves applied so far
    replay = board.copy()                                 # validate the whole line first

    try:
        for uci in solution:
            replay.push_uci(uci)
    except ValueError:
        return []

    for uci in solution:
        if board.turn == solver:
            k = k + 1
            n_remaining = n_total - k + 1
            legal = build_legal_moves(board)
            x = build_node_features(board)
            edge_index, edge_attr = build_edge(board)
            edge_time = None
            t_value = None
            if use_timing:
                t_value = simulate_think_time(
                    row['Rating'], n_remaining, len(legal))
                t = np.full((64, 1), t_value, dtype=np.float32)
                x = np.concatenate([x, t], axis=1)
                edge_time = build_edge_time(edge_index, last_moved, played)
            examples.append({
                # what the network reads
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "edge_time": edge_time,
                # what it has to predict
                "y": build_label([uci]),
                "legal_moves": legal,
                "think_time": t_value,
                # bookkeeping: for analysis, not for the network
                "n_remaining": n_total - k + 1,   # true depth of THIS position
                "puzzle_n": n_total,              # depth of the whole puzzle
                "puzzle_id": row.get("PuzzleId"),  # to check for split leakage
                "rating": row["Rating"],
                "fen": board.fen(),               # to rebuild the position later
                "solution": solution,             # the script, used at eval time
            })

            if not unroll:
                break

        move = chess.Move.from_uci(uci)
        board.push(move)
        played = played + 1
        update_last_moved(last_moved, move, played)

    return examples


def build_edge_time(edge_index, last_moved, played):
    """Recency of every edge, as an array of length E (in plies).

    An edge is generated by the piece on its source square, so it inherits that
    piece's recency: how many plies ago it last moved, 0 meaning "just moved".
    Pieces with no known history get UNKNOWN_RECENCY. The value must be scaled
    against the model's lambda_decay: plies run 0..UNKNOWN_RECENCY, so a lambda
    around 0.1-0.2 fits (see graph_dataset for the time_divisor trade-off).
    """
    times = np.empty(edge_index.shape[1], dtype=np.float32)
    for i in range(edge_index.shape[1]):
        src = int(edge_index[0, i])
        if src in last_moved:
            times[i] = played - last_moved[src]
        else:
            times[i] = UNKNOWN_RECENCY
    return times


def update_last_moved(last_moved, move, played):
    """Record that the piece now on move.to_square last moved at ply `played`."""
    last_moved.pop(move.from_square, None)
    last_moved[move.to_square] = played