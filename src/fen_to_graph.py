
# WHY THIS EXISTS
# The project compares a GNN against an LLM on mate-in-n puzzles.
# The LLM gets the FEN as text and must infer piece relationships itself.
# This encoder hands the GNN those relationships explicitly, as typed edges.

##Test:
##000Zo,4r3/1k6/pp3r2/1b2P2p/3R1p2/P1R2P2/1P4PP/6K1 w - - 0 35,e5f6 e8e1 g1f2 e1f1,1363,76,86,655,endgame mate mateIn2 operaMate short,https://lichess.org/n8Ff742v#69,,

import chess
import numpy as np

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
EDGE_TYPES = ["attacks", "defends", "moves","pushes"] #for a pawns a moves edge means control, 
"""
INPUT: chess.board,the pawn's square
OUTPUT:list of squares the pawn can advance to 

Legal move only covers the sive to move and we want to be able the pushes of both colours.


"""
def pawn_pushes(board,square,piece):
    if piece.color==chess.WHITE:    
        step=8
        start_rank=1
    else:
        step=-8
        start_rank=6
    targets=[]
    one=square+step
    if 0<=one<64 and board.piece_at(one) is None:
        targets.append(one)
        two=square+2*step
        if chess.square_rank(square)==start_rank and 0 <=two<64  and board.piece_at(two) is None:
            targets.append(two)

    return targets


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
def simulate_think_time(rating,n_remaining,n_legal):
    base = 2.0 + 18.0 * (rating - 400) / 2600.0    # ~2..20 s across the rating range
    depth = 1.0 + 0.35 * (n_remaining - 1)         # the first move takes the longest
    branch = 1.0 + 0.02 * (n_legal - 20)           # more options, more time to discard
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
    # We only keep queen promotions. Drop puzzles where the solver under-promotes.
    if any(len(m) == 5 and m[4] in "rbn" for m in solution[::2]):
        return []
    

    played=1
    last_moved={}
    update_last_moved(last_moved,chess.Move.from_uci(moves[0]),played)

    solver=board.turn #who moves now is the solver

    examples=[]
    k=0 #how many solver moves have been applied
    replay= board.copy() # we need to validate the line first
    try:
        for uci in solution:
            replay.push_uci(uci)
    except ValueError:
        return []
    for uci in solution: 
        if board.turn == solver:
            k=k+1
            n_remaining=n_total-k+1
            legal=build_legal_moves(board)
            x=build_node_features(board)
            edge_index, edge_attr = build_edge(board)
            edge_time= None 
            t_value=None
            if use_timing:
                t_value=simulate_think_time(row['Rating'],n_remaining,len(legal))
                t = np.full((64, 1), t_value, dtype=np.float32)
                x = np.concatenate([x, t], axis=1)
                edge_time=build_edge_time(edge_index,last_moved,played)
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

        move=chess.Move.from_uci(uci)
        board.push(move)
        played=played+1
        update_last_moved(last_moved,move,played)
        
 
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

UNKNOWN_RECENCY=20 #moved a long time ago, or never moved.


def build_edge_time(edge_index, last_moved, played):
    times=np.empty(edge_index.shape[1], dtype=np.float32)
    for i in range(edge_index.shape[1]):
        src=int(edge_index[0,i])
        if src in last_moved:
            times[i]=played-last_moved[src]
        else:
            times[i]= UNKNOWN_RECENCY
    return times


"""
INPUT:the move being played
OUTPUT:


"""
def update_last_moved(last_moved, move,played):
    last_moved.pop(move.from_square,None)
    last_moved[move.to_square]= played

TEST_ROW = {
    "PuzzleId": "000Zo",
    "FEN": "4r3/1k6/pp3r2/1b2P2p/3R1p2/P1R2P2/1P4PP/6K1 w - - 0 35",
    "Moves": "e5f6 e8e1 g1f2 e1f1",
    "Rating": "1363",
    "Themes": "endgame mate mateIn2 operaMate short",
    "MateIn": 2,
}
 
RESULTS = []
 
 
def check(name, condition, extra=""):
    RESULTS.append(bool(condition))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}"
          f"{'  ' + str(extra) if extra != '' else ''}")
 
 
def run_tests():
 
    # ---------------- normalize_row ----------------
    print("\n--- normalize_row ---")
    row = normalize_row(TEST_ROW)
    check("Moves became a list", isinstance(row["Moves"], list), row["Moves"])
    check("Themes became a list", isinstance(row["Themes"], list))
    check("Rating became an int", row["Rating"] == 1363)
    check("the caller's row was not modified", isinstance(TEST_ROW["Moves"], str))
    check("calling it twice is safe", normalize_row(row)["Moves"] == row["Moves"])
 
    # ---------------- get_puzzle_position ----------------
    print("\n--- get_puzzle_position ---")
    board, solution = get_puzzle_position(row)
    check("a board was returned", board is not None)
    check("the opponent move was applied",
          board.piece_at(chess.F6) is not None and board.piece_at(chess.E5) is None)
    check("BLACK is to move, not white as in the FEN", board.turn == chess.BLACK)
    check("solution is the remaining 3 moves", solution == ["e8e1", "g1f2", "e1f1"])
    replay = board.copy()
    for m in solution:
        replay.push_uci(m)
    check("the solution really is mate", replay.is_checkmate())
    b_none, s_none = get_puzzle_position({"FEN": "not a fen", "Moves": ["e2e4"]})
    check("invalid FEN returns (None, None)", b_none is None and s_none is None)
 
    # ---------------- build_node_features ----------------
    print("\n--- build_node_features ---")
    x = build_node_features(board)
    check("shape is (64, 12)", x.shape == (64, len(NODE_FEATURES)), x.shape)
    check("empty squares + pieces = 64",
          int(x[:, 8].sum()) + len(board.piece_map()) == 64)
    check("e8 is a black rook", x[chess.E8, 3] == 1.0 and x[chess.E8, 7] == 1.0)
    check("g1 is a white king", x[chess.G1, 5] == 1.0 and x[chess.G1, 6] == 1.0)
    check("a1 is empty", x[chess.A1, 8] == 1.0 and x[chess.A1, 0:6].sum() == 0.0)
    check("file matches python-chess",
          all(abs(x[s, 9] - chess.square_file(s) / 7.0) < 1e-6 for s in range(64)))
    check("rank matches python-chess",
          all(abs(x[s, 10] - chess.square_rank(s) / 7.0) < 1e-6 for s in range(64)))
    check("white_to_move is 0 everywhere (black to move)", x[:, 11].sum() == 0.0)
    check("every piece has one type and one colour flag",
          all(x[s, 0:6].sum() == 1.0 and x[s, 6:8].sum() == 1.0
              for s in board.piece_map()))
 
    # ---------------- pawn_pushes ----------------
    print("\n--- pawn_pushes ---")
    start = chess.Board()
    check("e2 pawn can go to e3 and e4",
          pawn_pushes(start, chess.E2, start.piece_at(chess.E2)) == [chess.E3, chess.E4])
    check("e7 black pawn can go to e6 and e5",
          pawn_pushes(start, chess.E7, start.piece_at(chess.E7)) == [chess.E6, chess.E5])
    check("a blocked pawn has no push",
          pawn_pushes(start, chess.A1, chess.Piece(chess.PAWN, chess.WHITE)) == [])
    check("f6 pawn in the puzzle can push to f7",
          pawn_pushes(board, chess.F6, board.piece_at(chess.F6)) == [chess.F7])
    check("a pawn off the start rank gets a single step",
          len(pawn_pushes(board, chess.F6, board.piece_at(chess.F6))) == 1)
 
    # ---------------- build_edge ----------------
    print("\n--- build_edge ---")
    edge_index, edge_attr = build_edge(board)
    E = edge_index.shape[1]
    check("edge_index has 2 rows", edge_index.shape[0] == 2, edge_index.shape)
    check("edge_attr has 4 columns now", edge_attr.shape == (E, 4), edge_attr.shape)
    check("every edge has exactly one type", np.all(edge_attr.sum(axis=1) == 1.0))
    check("all square indices are in 0..63",
          edge_index.min() >= 0 and edge_index.max() <= 63)
    check("no edge starts from an empty square",
          all(board.piece_at(int(s)) is not None for s in edge_index[0]))
    check("the b5 -> f1 edge exists (the mate rests on it)",
          any(edge_index[0, i] == chess.B5 and edge_index[1, i] == chess.F1
              for i in range(E)))
    check("f6 -> f7 exists and is typed 'pushes'",
          any(edge_index[0, i] == chess.F6 and edge_index[1, i] == chess.F7
              and edge_attr[i].argmax() == 3 for i in range(E)))
    check("d4 -> f4 is typed 'attacks'",
          any(edge_index[0, i] == chess.D4 and edge_index[1, i] == chess.F4
              and edge_attr[i].argmax() == 0 for i in range(E)))
    check("g1 -> g2 is typed 'defends'",
          any(edge_index[0, i] == chess.G1 and edge_index[1, i] == chess.G2
              and edge_attr[i].argmax() == 1 for i in range(E)))
    # no duplicated push edges: the pawn block must sit OUTSIDE the attacks loop
    pushes = [(int(edge_index[0, i]), int(edge_index[1, i]))
              for i in range(E) if edge_attr[i].argmax() == 3]
    check("no duplicated push edges", len(pushes) == len(set(pushes)), pushes)
    # cross-check every non-push edge against the board
    ok = True
    for i in range(E):
        if edge_attr[i].argmax() == 3:
            continue
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        occ = board.piece_at(dst)
        want = 2 if occ is None else (1 if occ.color == board.piece_at(src).color else 0)
        ok = ok and edge_attr[i].argmax() == want
    check("every edge type matches the board", ok)
    # the starting position has 16 pawns, each with 2 pushes
    ei0, ea0 = build_edge(chess.Board())
    check("starting position has 32 push edges", int(ea0[:, 3].sum()) == 32,
          int(ea0[:, 3].sum()))
 
    # ---------------- build_label / build_legal_moves ----------------
    print("\n--- build_label / build_legal_moves ---")
    check("e8e1 encodes to 3844", build_label(["e8e1"]) == 3844)
    check("decoding gives back e8 and e1", 3844 // 64 == chess.E8 and 3844 % 64 == chess.E1)
    legal = build_legal_moves(board)
    check("legal moves are far fewer than 4096", 0 < len(legal) < 100, len(legal))
    check("the correct move is among the legal ones", 3844 in legal)
    check("no duplicates in legal moves", len(legal) == len(set(legal.tolist())))
 
    # ---------------- row_to_graph ----------------
    print("\n--- row_to_graph ---")
    for flag in (False, True):
        ex = row_to_graph(TEST_ROW, use_timing=flag)
        check(f"use_timing={flag}: 2 examples", len(ex) == 2, len(ex))
        check(f"use_timing={flag}: x has {13 if flag else 12} features",
              ex[0]["x"].shape == (64, 13 if flag else 12), ex[0]["x"].shape)
 
    ex = row_to_graph(TEST_ROW)
    check("labels are 3844 then 261", [e["y"] for e in ex] == [3844, 261])
    check("y is always among the legal moves",
          all(e["y"] in e["legal_moves"] for e in ex))
    check("n_remaining goes 2 then 1", [e["n_remaining"] for e in ex] == [2, 1])
    check("n_remaining is an int", isinstance(ex[0]["n_remaining"], int),
          type(ex[0]["n_remaining"]).__name__)
    check("puzzle_n is an int", isinstance(ex[0]["puzzle_n"], int))
    check("edge_attr is aligned with edge_index",
          all(e["edge_attr"].shape[0] == e["edge_index"].shape[1] for e in ex))
    check("unroll=False gives a single example",
          len(row_to_graph(TEST_ROW, unroll=False)) == 1)
    check("an inconsistent MateIn drops the row",
          len(row_to_graph({**TEST_ROW, "MateIn": 3})) == 0)
    check("MateIn=-1 is accepted", len(row_to_graph({**TEST_ROW, "MateIn": -1})) == 2)
    check("MateIn=NaN is accepted",
          len(row_to_graph({**TEST_ROW, "MateIn": float("nan")})) == 2)
    check("a broken FEN drops the row",
          len(row_to_graph({**TEST_ROW, "FEN": "nope"})) == 0)
    print("\n--- corrupt line handling ---")
    for name, mv in [("malformed move mid-line", "e5f6 e8e1 zzzz e1f1"),
                     ("illegal move mid-line",   "e5f6 e8e1 a1a2 e1f1"),
                     ("illegal move at the end", "e5f6 e8e1 g1f2 h8h1")]:
        check(f"{name} drops the row instead of raising",
              len(row_to_graph({**TEST_ROW, "Moves": mv})) == 0)   
	    # ---------------- edge recency ----------------
    print("\n--- edge recency ---")

    def sources_with(ex, value):
        ei, et = ex["edge_index"], ex["edge_time"]
        return sorted({chess.square_name(int(ei[0, j]))
                       for j in range(ei.shape[1]) if et[j] == value})

    off = row_to_graph(TEST_ROW)
    on = row_to_graph(TEST_ROW, use_timing=True)

    check("edge_time is None when timing is off",
          all(e["edge_time"] is None for e in off))
    check("edge_time is set when timing is on",
          all(e["edge_time"] is not None for e in on))
    check("edge_time has one value per edge",
          all(len(e["edge_time"]) == e["edge_index"].shape[1] for e in on))
    check("no negative recency", all((e["edge_time"] >= 0).all() for e in on))
    check("recency never exceeds the unknown default",
          all((e["edge_time"] <= UNKNOWN_RECENCY).all() for e in on))

    check("example 1: the only piece with a history is f6, the setup move",
          sources_with(on[0], 0) == ["f6"])
    check("example 1: f6 generates 3 edges",
          int((on[0]["edge_time"] == 0).sum()) == 3)
    check("example 1: everything else is the default",
          set(on[0]["edge_time"].tolist()) == {0.0, float(UNKNOWN_RECENCY)})

    check("example 2: four recency levels",
          sorted(set(on[1]["edge_time"].tolist())) == [0.0, 1.0, 2.0, float(UNKNOWN_RECENCY)])
    check("example 2: the just-moved piece is the king on f2",
          sources_with(on[1], 0) == ["f2"])
    check("example 2: one ply earlier is the rook on e1",
          sources_with(on[1], 1) == ["e1"])
    check("example 2: two plies earlier is the pawn on f6",
          sources_with(on[1], 2) == ["f6"])
    check("the departed square e8 no longer carries a recency",
          "e8" not in sources_with(on[1], 0) + sources_with(on[1], 1) + sources_with(on[1], 2))
    check("test set (unroll=False) still carries edge_time",
          row_to_graph(TEST_ROW, use_timing=True, unroll=False)[0]["edge_time"] is not None)	
 
    # it must generalise to any depth, not just mate in 2
    b = chess.Board()
    seq = []
    for _ in range(10):
        m = next(iter(b.legal_moves))
        seq.append(m.uci())
        b.push(m)
    for n in (1, 2, 3, 5):
        r = {"PuzzleId": "x", "FEN": chess.STARTING_FEN, "Moves": " ".join(seq[:2 * n]),
             "Rating": 1000, "Themes": "mate", "MateIn": n}
        res = row_to_graph(r)
        check(f"n={n} gives {n} examples with decreasing depth",
              len(res) == n and [e["n_remaining"] for e in res] == list(range(n, 0, -1)))
 
    # ---------------- summary ----------------
    print("\n--- summary ---")
    ei, ea = build_edge(board)
    counts = {name: int(ea[:, t].sum()) for t, name in enumerate(EDGE_TYPES)}
    print(f"nodes    : 64 x {len(NODE_FEATURES)} features")
    print(f"edges    : {ei.shape[1]}   {counts}")
    print(f"position : {board.fen()}")
    print(f"solution : {' '.join(solution)}")
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    print("everything works" if all(RESULTS)
          else "something is broken, look at the FAIL lines above")
 
if __name__ == "__main__":
    run_tests()
