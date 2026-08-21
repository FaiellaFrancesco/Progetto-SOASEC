##Test:
##000Zo,4r3/1k6/pp3r2/1b2P2p/3R1p2/P1R2P2/1P4PP/6K1 w - - 0 35,e5f6 e8e1 g1f2 e1f1,1363,76,86,655,endgame mate mateIn2 operaMate short,https://lichess.org/n8Ff742v#69,,

## This is used to build a graph of the chess positions and their relationships
## INPUT: lichess_db_puzzle.csv filtered with fen, moves, rating, themes 
## OUTPUT: graph.json with nodes and edges representing the chess positions and their relationships

import csv
import json
import chess
import numpy as np


"""

"""

def normalize_row(row):
    # Normalize the row 
    r=dict(row)
    if isinstance(r["Moves"], str):
        r["Moves"]=r["Moves"].split()
    if isinstance(r["Themes"], str):
        r["Themes"]=r["Themes"].split()
    if isinstance(r["Rating"], str):
        r["Rating"]=int(r["Rating"])
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
        board.push_uci(row["Moves"][0])  # Apply the first move to get the puzzle position
        return board,row["Moves"][1:]  # Return the board and the remaining moves
    except ValueError:
        print(f"Invalid FEN: {row['FEN']}")
        return None,None

NODE_FEATURES = ["pawn", "knight", "bishop", "rook", "queen", "king", "white", "black", "empty", "file", "rank", "white_to_move"]
PIECE_TYPES= [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
EDGE_TYPES = ["attacks", "defends", "moves"]


"""
INPUT: chess.Board object representing the puzzle position
OUTPUT: numpy array of shape (64, 12) one row per square
Every square of the board is a node numbered from a1=0, b1=1, ..., h8=63. Each node is described by 12 numbers: which piece stands on it,
which color it is, whather it is empty, where where it sits on the board, and whose turn it is to move.

"""

def build_node_features(board):
    x= np.zeros((64, len(NODE_FEATURES)), dtype=np.float32)

    for square in range(64):
        piece=board.piece_at(square)
        if piece is None:
            x[square,8] = 1.0 # empty (no piece in that square)
        else:
            x[square,PIECE_TYPES.index(piece.piece_type)] = 1.0 # which piece type is on that square
            if piece.color == chess.WHITE:
                x[square,6] = 1.0 # white piece
            else:
                x[square,7] = 1.0 # black piece

        x[square,9] = square % 8 / 7.0 # file (column) of the square, normalized to [0,1]
        x[square,10] = square // 8 / 7.0 # rank (row) of the square, normalized to [0,1]
        if board.turn == chess.WHITE:
            x[square,11] = 1.0 # white to move
        else:
            x[square,11] = 0.0 # black to move
    return x

"""
INPUT:chess.board object
OUTPUT: edge_index, numpy array of shape (2, E) 
        edge_attr, numpy array of shape (E, 3) 
The edges are directed and represent the relationships between the pieces on the board.ù

"""

def build_edge(board):
    sources, targets, kinds = [], [], []
    for square in range(64):
        piece=board.piece_at(square)
        if piece is None:
            continue # empty square
        for target in board.attacks(square):
            occupant=board.piece_at(target)

            if occupant is None:
                kind=2 #moves - empty square
            elif occupant.color == piece.color:
                    kind=1 #defends - same color
            else:
                    kind=0 #attacks - different color
            sources.append(square)
            targets.append(target)
            kinds.append(kind)
                 # 2 x num_edges: row 0 = source square, row 1 = target square
    edge_index = np.array([sources, targets], dtype=np.int64)
 
    # num_edges x 3: one-hot of the relation type
    edge_attr = np.zeros((len(kinds), len(EDGE_TYPES)), dtype=np.float32)
    edge_attr[np.arange(len(kinds)), kinds] = 1.0
 
    return edge_index, edge_attr

######################################################################

"""
fen_to_graph.py  -  turn a Lichess puzzle row into a graph.

BOUNDARY WITH MY TEAMMATE
-------------------------
He downloads the CSV, filters mateIn1..5 and hands me the rows.
I start from a row that is already parsed: a dict or a pandas row, with

    FEN     str
    Moves   list of UCI moves (or a space-separated string)
    Rating  number
    Themes  list (or a space-separated string)

I do NOT read the CSV.

AGREED: the opponent move (Moves[0]) is applied by ME, not by him.
He hands me the row exactly as it appears in the CSV, untouched.
"""

import chess
import numpy as np


# ---------------------------------------------------------------
# input
# ---------------------------------------------------------------
def normalize_row(row):
    """
    Airlock. Accept Moves and Themes either as a list or as a string,
    because pandas.read_csv hands them over as strings.

    Why it matters: if Moves is the string "e5f6 e8e1 g1f2 e1f1", then
    Moves[0] is the CHARACTER "e", not the opponent move.

    Works on a copy so the caller's row is never mutated, and it is
    idempotent: calling it on an already-clean row changes nothing.
    """
    r = dict(row)
    if isinstance(r["Moves"], str):
        r["Moves"] = r["Moves"].split()
    if isinstance(r.get("Themes"), str):
        r["Themes"] = r["Themes"].split()
    r["Rating"] = int(r["Rating"])
    return r


def get_puzzle_position(row):
    """
    In the Lichess dataset the FEN is NOT the puzzle position: it is the
    position one ply earlier. Moves[0] is the OPPONENT's move. Applying it
    is what gives the real starting position; the remaining moves are the
    solution.

    push_uci() also validates legality, so a corrupted row raises here.
    Note that it mutates `board` in place.
    """
    board = chess.Board(row["FEN"])
    board.push_uci(row["Moves"][0])
    return board, row["Moves"][1:]


# ---------------------------------------------------------------
# nodes  ->  x
# ---------------------------------------------------------------
# 64 nodes, one per square, numbered a1=0, b1=1, ..., h8=63.
NODE_FEATURES = ["pawn", "knight", "bishop", "rook", "queen", "king",
                 "white", "black", "empty", "file", "rank", "white_to_move"]

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]


def build_node_features(board):
    x = np.zeros((64, len(NODE_FEATURES)), dtype=np.float32)

    for square in range(64):
        piece = board.piece_at(square)

        if piece is None:
            x[square, 8] = 1.0                                          # empty
        else:
            x[square, PIECE_TYPES.index(piece.piece_type)] = 1.0        # which piece
            x[square, 6 if piece.color == chess.WHITE else 7] = 1.0     # which colour

        x[square, 9] = chess.square_file(square) / 7.0
        x[square, 10] = chess.square_rank(square) / 7.0
        x[square, 11] = 1.0 if board.turn == chess.WHITE else 0.0       # global flag

    return x


# ---------------------------------------------------------------
# edges  ->  edge_index + edge_attr
# ---------------------------------------------------------------
# board.attacks(square) returns the squares controlled by the piece standing
# there. python-chess already knows how pieces move, so the chess rules are
# not reimplemented here.
#
# TODO: pawns only control the two diagonals, so pawn PUSHES do not show up
# as edges. Add a fourth edge type "pushes".
EDGE_TYPES = ["attacks", "defends", "moves"]


def build_edges(board):
    sources, targets, kinds = [], [], []

    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            continue                        # an empty square emits no edges

        for target in board.attacks(square):
            occupant = board.piece_at(target)

            if occupant is None:
                kind = 2                    # moves   -> empty square
            elif occupant.color != piece.color:
                kind = 0                    # attacks -> enemy piece
            else:
                kind = 1                    # defends -> friendly piece

            sources.append(square)
            targets.append(target)
            kinds.append(kind)

    # 2 x num_edges: row 0 = source square, row 1 = target square
    edge_index = np.array([sources, targets], dtype=np.int64)

    # num_edges x 3: one-hot of the relation type
    edge_attr = np.zeros((len(kinds), len(EDGE_TYPES)), dtype=np.float32)
    edge_attr[np.arange(len(kinds)), kinds] = 1.0

    return edge_index, edge_attr


# ---------------------------------------------------------------
# label  ->  y
# ---------------------------------------------------------------
def build_label(solution):
    """
    The model has to predict the FIRST move of the solution, encoded as a
    single integer in [0, 4095]: from_square * 64 + to_square.
    """
    move = chess.Move.from_uci(solution[0])
    return move.from_square * 64 + move.to_square


# ---------------------------------------------------------------
# public API
# ---------------------------------------------------------------
def row_to_graph(row, use_timing=False):
    """Teammate's row -> graph ready for the network."""
    r = normalize_row(row)
    board, solution = get_puzzle_position(r)

    x = build_node_features(board)
    edge_index, edge_attr = build_edges(board)
    y = build_label(solution)

    if use_timing:
        # ABLATION STUDY SLOT (project objective 5).
        # Placeholder thinking time derived from the rating: harder puzzle =
        # longer think time. Swap in real per-move times from the PGNs later
        # without touching anything else.
        time_feat = np.full((64, 1), r["Rating"] / 3000.0, dtype=np.float32)
        x = np.concatenate([x, time_feat], axis=1)

    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr,
            "y": y, "board": board, "solution": solution, "row": r}


def verify_mate(g):
    """Quality check: does the solution actually deliver mate?"""
    b = g["board"].copy()          # copy: push() mutates the board in place
    for move in g["solution"]:
        b.push_uci(move)
    return b.is_checkmate()


def rows_to_graphs(rows, use_timing=False, drop_non_mate=True):
    """
    Many rows -> many graphs. Broken rows are skipped instead of crashing
    the whole preprocessing halfway through.
    """
    graphs, skipped = [], 0
    for row in rows:
        try:
            g = row_to_graph(row, use_timing=use_timing)
            if drop_non_mate and not verify_mate(g):
                skipped += 1
                continue
            graphs.append(g)
        except Exception:
            skipped += 1
    return graphs, skipped


# ---------------------------------------------------------------
# demo
# ---------------------------------------------------------------
if __name__ == "__main__":

    # this row arrives already parsed from my teammate
    row = {
        "PuzzleId": "000Zo",
        "FEN": "4r3/1k6/pp3r2/1b2P2p/3R1p2/P1R2P2/1P4PP/6K1 w - - 0 35",
        "Moves": "e5f6 e8e1 g1f2 e1f1",
        "Rating": 1363,
        "Themes": "endgame mate mateIn2 operaMate short",
    }

    g = row_to_graph(row, use_timing=True)
    board = g["board"]

    print("=== real puzzle position ===")
    print("FEN received     :", row["FEN"])
    print("opponent move    :", g["row"]["Moves"][0], " <- applied by me")
    print("puzzle position  :", board.fen())
    print("side to move     :", "white" if board.turn else "BLACK")
    print("solution         :", g["solution"])
    print()
    print(board)          # UPPERCASE = white, lowercase = black, . = empty

    print("\n=== nodes ===")
    print("x:", g["x"].shape, "-> 64 squares x", g["x"].shape[1], "features")
    for square in [chess.E8, chess.B5, chess.G1]:
        vals = " ".join(f"{v:.2f}" for v in g["x"][square])
        print(f"  {chess.square_name(square):>3} (node {square:2d}): {vals}")
    print("  ", NODE_FEATURES + ["time"])

    print("\n=== edges ===")
    ei, ea = g["edge_index"], g["edge_attr"]
    print("edge_index:", ei.shape, "->", ei.shape[1], "edges")
    print("edge_attr :", ea.shape)
    for t, name in enumerate(EDGE_TYPES):
        print(f"  {name:8s}: {int(ea[:, t].sum())}")

    print("\n  the edge the whole mate rests on:")
    for i in range(ei.shape[1]):
        if ei[0, i] == chess.B5 and ei[1, i] == chess.F1:
            print("    b5 -> f1  (", EDGE_TYPES[int(ea[i].argmax())], ")")

    print("\n=== label ===")
    m = chess.Move.from_uci(g["solution"][0])
    print(f"move to predict: {m.uci()} = {chess.square_name(m.from_square)}"
          f" -> {chess.square_name(m.to_square)}")
    print(f"y = {g['y']}  ({m.from_square} * 64 + {m.to_square})")

    print("\n=== quality check ===")
    print("solution delivers mate:", verify_mate(g))

    graphs, skipped = rows_to_graphs([row, row])
    print(f"batch test: {len(graphs)} valid graphs, {skipped} skipped")