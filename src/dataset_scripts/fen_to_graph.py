# WHY THIS EXISTS
# The project compares a GNN against an LLM on mate-in-n puzzles.
# The LLM gets the FEN as text and must infer piece relationships itself.
# This encoder hands the GNN those relationships explicitly, as typed edges.

# Test:
# 000Zo,4r3/1k6/pp3r2/1b2P2p/3R1p2/P1R2P2/1P4PP/6K1 w - - 0 35,e5f6 e8e1 g1f2 e1f1,1363,76,86,655,endgame mate mateIn2 operaMate short,https://lichess.org/n8Ff742v#69,,

import chess
import numpy as np

NODE_FEATURES = ["pawn", "knight", "bishop", "rook", "queen",
                 "king", "white", "black", "empty", "file", "rank", "white_to_move"]
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK,
               chess.QUEEN, chess.KING]  # for a pawns a moves edge means control,
EDGE_TYPES = ["attacks", "defends", "moves", "pushes"]

UNKNOWN_RECENCY = 20  # moved a long time ago, or never moved.


def normalize_row(row):
    # Normalize the row
    r = dict(row)
    if isinstance(r["Moves"], str):
        r["Moves"] = r["Moves"].split()
    if isinstance(r["Themes"], str):
        r["Themes"] = r["Themes"].split()
    if isinstance(r["Rating"], str):
        r["Rating"] = int(r["Rating"])
    return r


"""
INPUT: row from the CSV file
OUTPUT: chess.Board object representing the position of the puzzle, and the list of moves 

We check if the FEN string is valid, and if it is we create a chess.Board object 
and apply the first move to get the position of the puzzle.
"""


def get_puzzle_position(row):
    # The problem is that the FEN string in the CSV file may not be valid, so we need to check if it is valid before creating a chess.Board object
    # Also the FEN is one Move before the puzzle, so we need to apply the moves to the board to get the position of the puzzle
    try:
        board = chess.Board(row["FEN"])
        # Apply the first move to get the puzzle position
        board.push_uci(row["Moves"][0])
        # Return the board and the remaining moves
        return board, row["Moves"][1:]
    except ValueError:
        print(f"Invalid FEN: {row['FEN']}")
        return None, None


"""
INPUT: chess.board,the pawn's square
OUTPUT:list of squares the pawn can advance to 

Legal move only covers the sive to move and we want to be able the pushes of both colours.


"""


def pawn_pushes(board, square, piece):
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


"""
INPUT: chess.Board object representing the puzzle position
OUTPUT: numpy array of shape (64, 12) one row per square
Every square of the board is a node numbered from a1=0, b1=1, ..., h8=63. Each node is described by 12 numbers: which piece stands on it,
which color it is, whather it is empty, where where it sits on the board, and whose turn it is to move.

"""


def build_node_features(board):
    x = np.zeros((64, len(NODE_FEATURES)), dtype=np.float32)

    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            x[square, 8] = 1.0  # empty (no piece in that square)
        else:
            # which piece type is on that square
            x[square, PIECE_TYPES.index(piece.piece_type)] = 1.0
            if piece.color == chess.WHITE:
                x[square, 6] = 1.0  # white piece
            else:
                x[square, 7] = 1.0  # black piece

        # file (column) of the square, normalized to [0,1]
        x[square, 9] = square % 8 / 7.0
        # rank (row) of the square, normalized to [0,1]
        x[square, 10] = square // 8 / 7.0
        if board.turn == chess.WHITE:
            x[square, 11] = 1.0  # white to move
        else:
            x[square, 11] = 0.0  # black to move
    return x


"""
INPUT:chess.board object
OUTPUT: edge_index, numpy array of shape (2, E) 
        edge_attr, numpy array of shape (E, 4) 
The edges are directed and represent the relationships between the pieces on the board.

"""


def build_edge(board):
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


"""
INPUT:List of moves
OUTPUT:List of moves encoded as a score

example the move "e8e1" = from 60 to  4 in y= 3844
in fact:
3844 //64 = 60 -> e8 and
3844 % 64 = 4  -> e1



"""


def build_label(solution):
    move = chess.Move.from_uci(solution[0])
    return move.from_square * 64 + move.to_square


"""
INPUT: y (the class the model predicted, 0..4095), board (the position y refers to)
OUTPUT: a chess.Move object, ready for board.push()

build_label goes move -> integer, decode_move goes integer -> move: they are a
PAIR and have to stay consistent.

The model has 4096 classes, one per (from, to) pair, so it cannot say WHICH
piece a pawn promotes to. build_label drops that information, so decode_move
puts it back with the same convention the encoder uses: a pawn landing on rank
1 or rank 8 always becomes a queen. Puzzles whose solution needs an
under-promotion are already dropped by row_to_graph, so the two agree.

The board is needed exactly for this check: without it we cannot tell whether
the piece standing on the source square is a pawn.
"""


def decode_move(y, board):
    src = y // 64
    dst = y % 64
    piece = board.piece_at(src)
    if (piece is not None and piece.piece_type == chess.PAWN
            and chess.square_rank(dst) in (0, 7)):
        return chess.Move(src, dst, promotion=chess.QUEEN)
    return chess.Move(src, dst)


"""
INPUT: chess.Board object
OUTPUT: numpy array of the legal moves, encoded like y


the model doesnt know the rules of chess 
"""


def build_legal_moves(board):
    idx = [m.from_square * 64 + m.to_square for m in board.legal_moves]
    return np.unique(np.array(idx, dtype=np.int16))


"""
INPUT: rating of the puzzle, n_remaining (solver moves left), n_legal (number of legal moves in this position)
OUTPUT: simulated think time, normalised to [0, 1]

2.0, 18.0, 0.35, 0.02 and 60.0 are hand-tuned values for plausible human think time.

"""


def simulate_think_time(rating, n_remaining, n_legal):
    # ~2..20 s across the rating range
    base = 2.0 + 18.0 * (rating - 400) / 2600.0
    # the first move takes the longest
    depth = 1.0 + 0.35 * (n_remaining - 1)
    # more options, more time to discard
    branch = 1.0 + 0.02 * (n_legal - 20)
    seconds = base * depth * branch
    return min(max(seconds / 60.0, 0.0), 1.0)


"""
Lichess only tags mateIn1..5, so a deeper mate has no tag 

INPUT: row from CSV, carrying MateIn colunm
    usetiming=
    unroll=         - True: one example per solver decision (train / val)
                    - False: only the first move (test set)

 
OUTPUT: a list of examples. Empty list the row is unusable

info:

in a matein puzzle the solver plays n move and the opponet n-1, so the solution is 2n-1 moves, and with the opponent's setup moves at index 0 len(Moves)=2n. Unrolling turn one puzzle in a n training puzzles, for example a "matein5" turns inmatein5, then matein4, matein3... 

y stays ONE move for examples, every later moves belongs to a different position that does not exist yet. The full move list is saved as metadata under "solution"

Time:



"""


def row_to_graph(row, use_timing=False, unroll=True):
    row = normalize_row(row)
    moves = row["Moves"]
    # total moves for the resolver to solve the puzzle
    n_total = len(moves) // 2
    if n_total < 1:
        return []
    try:
        tagged = int(row["MateIn"])
    except (KeyError, ValueError, TypeError):
        tagged = None
    if tagged is not None and tagged > 0 and tagged != n_total:
        return []  # skip puzzles that are tagged with a different mate-in than the number of moves in the puzzle

    board, solution = get_puzzle_position(row)
    if board is None:
        return []

    # We only keep queen promotions. Drop puzzles where the solver under-promotes.
    if any(len(m) == 5 and m[4] in "rbn" for m in solution[::2]):
        return []

    played = 1
    last_moved = {}
    update_last_moved(last_moved, chess.Move.from_uci(moves[0]), played)

    solver = board.turn  # who moves now is the solver

    examples = []
    k = 0  # how many solver moves have been applied
    replay = board.copy()  # we need to validate the line first

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


"""
ply: è la semi-mossa, quindi la mossa di un solo giocatore 
INPUT:board edge_idex (2, E), last_moved ()dict square --> ply it arrived on), played (quante semi mosse sono state giocate)
OUTPUT: numpy array of lenght E, the recency of every edge

##INFO

An edgy is generated by the piece standing on its SOURCE square, so the esge inherits that piece's recency : how many semi-mosse ago that piece last moved, 0 means "it's just moved"

in the puzzles there are peice with no know history (we could use the link to get the whole game, but for millions row is a lot) so we use the 
UNKNOWN_RECENCY.

The value fed to teh model must be scaled against its lamba_decay, The Plies (semi-mosse) range to 0..UNKOWN_RECENCY, so lambda around 0.1-02.
"""


def build_edge_time(edge_index, last_moved, played):
    times = np.empty(edge_index.shape[1], dtype=np.float32)
    for i in range(edge_index.shape[1]):
        src = int(edge_index[0, i])
        if src in last_moved:
            times[i] = played - last_moved[src]
        else:
            times[i] = UNKNOWN_RECENCY
    return times


"""
INPUT:the move being played
OUTPUT:


"""


def update_last_moved(last_moved, move, played):
    last_moved.pop(move.from_square, None)
    last_moved[move.to_square] = played
