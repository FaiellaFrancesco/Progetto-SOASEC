import chess
from fen_to_graph import *

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


def fen_to_graph_tests():

    # ---------------- normalize_row ----------------
    print("\n--- normalize_row ---")
    row = normalize_row(TEST_ROW)
    check("Moves became a list", isinstance(row["Moves"], list), row["Moves"])
    check("Themes became a list", isinstance(row["Themes"], list))
    check("Rating became an int", row["Rating"] == 1363)
    check("the caller's row was not modified",
          isinstance(TEST_ROW["Moves"], str))
    check("calling it twice is safe", normalize_row(
        row)["Moves"] == row["Moves"])

    # ---------------- get_puzzle_position ----------------
    print("\n--- get_puzzle_position ---")
    board, solution = get_puzzle_position(row)
    check("a board was returned", board is not None)
    check("the opponent move was applied",
          board.piece_at(chess.F6) is not None and board.piece_at(chess.E5) is None)
    check("BLACK is to move, not white as in the FEN", board.turn == chess.BLACK)
    check("solution is the remaining 3 moves",
          solution == ["e8e1", "g1f2", "e1f1"])
    replay = board.copy()
    for m in solution:
        replay.push_uci(m)
    check("the solution really is mate", replay.is_checkmate())
    b_none, s_none = get_puzzle_position(
        {"FEN": "not a fen", "Moves": ["e2e4"]})
    check("invalid FEN returns (None, None)",
          b_none is None and s_none is None)

    # ---------------- build_node_features ----------------
    print("\n--- build_node_features ---")
    x = build_node_features(board)
    check("shape is (64, 12)", x.shape == (64, len(NODE_FEATURES)), x.shape)
    check("empty squares + pieces = 64",
          int(x[:, 8].sum()) + len(board.piece_map()) == 64)
    check("e8 is a black rook", x[chess.E8, 3]
          == 1.0 and x[chess.E8, 7] == 1.0)
    check("g1 is a white king", x[chess.G1, 5]
          == 1.0 and x[chess.G1, 6] == 1.0)
    check("a1 is empty", x[chess.A1, 8] ==
          1.0 and x[chess.A1, 0:6].sum() == 0.0)
    check("file matches python-chess",
          all(abs(x[s, 9] - chess.square_file(s) / 7.0) < 1e-6 for s in range(64)))
    check("rank matches python-chess",
          all(abs(x[s, 10] - chess.square_rank(s) / 7.0) < 1e-6 for s in range(64)))
    check("white_to_move is 0 everywhere (black to move)",
          x[:, 11].sum() == 0.0)
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
    check("edge_attr has 4 columns now",
          edge_attr.shape == (E, 4), edge_attr.shape)
    check("every edge has exactly one type",
          np.all(edge_attr.sum(axis=1) == 1.0))
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
        want = 2 if occ is None else (
            1 if occ.color == board.piece_at(src).color else 0)
        ok = ok and edge_attr[i].argmax() == want
    check("every edge type matches the board", ok)
    # the starting position has 16 pawns, each with 2 pushes
    ei0, ea0 = build_edge(chess.Board())
    check("starting position has 32 push edges", int(ea0[:, 3].sum()) == 32,
          int(ea0[:, 3].sum()))

    # ---------------- build_label / build_legal_moves ----------------
    print("\n--- build_label / build_legal_moves ---")
    check("e8e1 encodes to 3844", build_label(["e8e1"]) == 3844)
    check("decoding gives back e8 and e1", 3844 //
          64 == chess.E8 and 3844 % 64 == chess.E1)
    legal = build_legal_moves(board)
    check("legal moves are far fewer than 4096",
          0 < len(legal) < 100, len(legal))
    check("the correct move is among the legal ones", 3844 in legal)
    check("no duplicates in legal moves", len(
        legal) == len(set(legal.tolist())))

    # ---------------- simulate_think_time ----------------
    print("\n--- simulate_think_time ---")
    check("always inside [0, 1]",
          all(0.0 <= simulate_think_time(r, d, l) <= 1.0
              for r in range(400, 3001, 200)
              for d in range(1, 11)
              for l in range(0, 60, 5)))
    check("a harder puzzle takes longer",
          simulate_think_time(2400, 1, 20) > simulate_think_time(800, 1, 20))
    check("more remaining depth takes longer",
          simulate_think_time(1500, 5, 20) > simulate_think_time(1500, 1, 20))
    check("more legal moves takes longer",
          simulate_think_time(1500, 1, 40) > simulate_think_time(1500, 1, 10))
    check("the extreme case is capped at 1",
          simulate_think_time(3000, 10, 60) == 1.0)

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
    check("n_remaining goes 2 then 1", [
          e["n_remaining"] for e in ex] == [2, 1])
    check("n_remaining is an int", isinstance(ex[0]["n_remaining"], int),
          type(ex[0]["n_remaining"]).__name__)
    check("puzzle_n is an int", isinstance(ex[0]["puzzle_n"], int))
    check("edge_attr is aligned with edge_index",
          all(e["edge_attr"].shape[0] == e["edge_index"].shape[1] for e in ex))
    check("unroll=False gives a single example",
          len(row_to_graph(TEST_ROW, unroll=False)) == 1)
    check("an inconsistent MateIn drops the row",
          len(row_to_graph({**TEST_ROW, "MateIn": 3})) == 0)
    check("MateIn=-1 is accepted",
          len(row_to_graph({**TEST_ROW, "MateIn": -1})) == 2)
    check("MateIn=NaN is accepted",
          len(row_to_graph({**TEST_ROW, "MateIn": float("nan")})) == 2)
    check("a broken FEN drops the row",
          len(row_to_graph({**TEST_ROW, "FEN": "nope"})) == 0)

    # ---------------- underpromotion guard ----------------
    print("\n--- underpromotion guard ---")
    UNDER = {"PuzzleId": "u", "FEN": "r6k/1P6/8/8/8/8/8/6K1 b - - 0 1",
             "Moves": "a8a7 b7b8n", "Rating": 1500, "Themes": "mate",
             "MateIn": 1}
    QUEEN = {**UNDER, "Moves": "a8a7 b7b8q"}
    OPP = {"PuzzleId": "o", "FEN": "7k/1P6/8/8/8/8/K7/7r w - - 0 1",
           "Moves": "b7b8n h1h2", "Rating": 1500, "Themes": "mate",
           "MateIn": 1}
    check("a solver underpromotion drops the row",
          len(row_to_graph(UNDER)) == 0)
    check("a queen promotion is kept", len(row_to_graph(QUEEN)) == 1)
    check("an opponent underpromotion is harmless",
          len(row_to_graph(OPP)) == 1)

    # ---------------- corrupt line handling ----------------
    print("\n--- corrupt line handling ---")
    for name, mv in [("malformed move mid-line", "e5f6 e8e1 zzzz e1f1"),
                     ("illegal move mid-line",   "e5f6 e8e1 a1a2 e1f1"),
                     ("illegal move at the end", "e5f6 e8e1 g1f2 h8h1")]:
        check(f"{name} drops the row instead of raising",
              len(row_to_graph({**TEST_ROW, "Moves": mv})) == 0)

    off = row_to_graph(TEST_ROW)
    on = row_to_graph(TEST_ROW, use_timing=True)

    # ---------------- simulated think time on the examples ----------------
    print("\n--- think time on the examples ---")
    check("think_time is None when timing is off",
          all(e["think_time"] is None for e in off))
    check("think_time is set when timing is on",
          all(e["think_time"] is not None for e in on))
    check("think_time matches the last column of x",
          all(abs(e["think_time"] - float(e["x"][0, -1])) < 1e-6 for e in on))
    check("think_time is inside [0, 1]",
          all(0 <= e["think_time"] <= 1 for e in on))
    check("think_time is constant across the 64 nodes",
          all(np.all(e["x"][:, -1] == e["x"][0, -1]) for e in on))
    check("think_time differs between the two examples",
          on[0]["think_time"] != on[1]["think_time"],
          [round(e["think_time"], 3) for e in on])
    check("think_time decreases as the mate gets closer",
          on[0]["think_time"] > on[1]["think_time"])

    # ---------------- edge recency ----------------
    print("\n--- edge recency ---")

    def sources_with(ex, value):
        ei, et = ex["edge_index"], ex["edge_time"]
        return sorted({chess.square_name(int(ei[0, j]))
                       for j in range(ei.shape[1]) if et[j] == value})

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

    # ---------------- decode_move ----------------
    print("\n--- decode_move ---")
    for name, fen, src, dst, want in [
            ("a white pawn reaching the 8th rank becomes a queen",
             "8/1P6/8/8/8/8/8/K6k w - - 0 1", chess.B7, chess.B8, "b7b8q"),
            ("a black pawn reaching the 1st rank becomes a queen",
             "K6k/8/8/8/8/8/1p6/8 b - - 0 1", chess.B2, chess.B1, "b2b1q"),
            ("a normal pawn move is left alone",
             "8/8/8/8/8/8/1P6/K6k w - - 0 1", chess.B2, chess.B4, "b2b4"),
            ("a rook reaching the 8th rank is NOT a promotion",
             "8/8/8/8/8/8/8/KR5k w - - 0 1", chess.B1, chess.B8, "b1b8"),
            ("a promotion by capture also becomes a queen",
             "1n6/P7/8/8/8/8/8/K6k w - - 0 1", chess.A7, chess.B8, "a7b8q")]:
        bb = chess.Board(fen)
        m = decode_move(src * 64 + dst, bb)
        check(name, m.uci() == want and m in bb.legal_moves, m.uci())

    check("decode_move inverts build_label on every legal opening move",
          all(decode_move(build_label([m.uci()]), start).uci()[:4] == m.uci()[:4]
              for m in start.legal_moves))
    check("y=3844 decodes back to e8e1",
          decode_move(3844, board).uci() == "e8e1")

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
    fen_to_graph_tests()
