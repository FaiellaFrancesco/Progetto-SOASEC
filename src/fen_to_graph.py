
# WHY THIS EXISTS
# The project compares a GNN against an LLM on mate-in-n puzzles.
# The LLM gets the FEN as text and must infer piece relationships itself.
# This encoder hands the GNN those relationships explicitly, as typed edges.

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

"""
INPUT:List of moves
OUTPUT:List of moves encoded as a score

example the move "e8e1" = from 60 to  4 in y= 3844
in fact:
3844 //64 = 60 -> e8 and
3864 % 64 = 4  -> e1



"""
def build_label(solution):
    move = chess.Move.from_uci(solution[0])
    return move.from_square * 64 + move.to_square
"""
INPUT: chess.Board object
OUTPUT: numpy array of the legal moves, encoded like y


the model doesnt know the rules of chess 
"""


def build_legal_moves(board):
    idx = [m.from_square * 64 + m.to_square for m in board.legal_moves]
    return np.unique(np.array(idx, dtype=np.int16))
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
"""

def row_to_graph(row,use_timing=False, unroll=True):
    row=normalize_row(row)
    moves=row["Moves"]
    n_total=len(moves)//2 #total moves for the resolver to solve the puzzle
    if n_total < 1:
        return []
    
    try:
        tagged=int(row["MateIn"])
    except (KeyError, ValueError, TypeError):
        tagged=None
    if tagged is not None and tagged > 0 and tagged != n_total:
        return [] #skip puzzles that are tagged with a different mate-in than the number of moves in the puzzle
    
    board, solution = get_puzzle_position(row)
    if board is None:
        return []    
    
    solver=board.turn #who moves now is the solver

    examples=[]
    k=0 #how many solver moves have been applied

    for uci in solution: 
        if board.turn == solver:
            k=k+1
            x=build_node_features(board)
            edge_index, edge_attr = build_edge(board)
            if use_timing:
                t = np.full((64, 1), row["Rating"] / 3000.0, dtype=np.float32)
                x = np.concatenate([x, t], axis=1)

            examples.append({
           # what the network reads
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            # what it has to predict
            "y": build_label([uci]),
            "legal_moves": build_legal_moves(board),
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

    
        board.push_uci(uci)
 
    return examples
