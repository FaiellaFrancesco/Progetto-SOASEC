#!/usr/bin/env bash
# valuta_tutto.sh
#
# Runs every test-set evaluation of the final models, in sequence, and saves
# each result under risultati/fix/ with its own name.
#
#     cd ~/Desktop/Progetto-SOASEC
#     caffeinate -i bash src/chess_eval/valuta_tutto.sh
#
# Three measurements x two test sets = six runs:
#
#   evaluation/evaluate_models.py   first move, McNemar between the two models
#   evaluation/eval_ranks.py        top-1/3/5 with min / expected / max over ties
#   evaluation/playout_models.py    whole puzzles, against the recorded defence
#
#   full test   graphs/test, graphs/test_plain, dataset/test.csv
#   clean test  the same minus the 973 positions also present in training
#
# Every run is also written to a log. If any step fails the script stops
# there: no result file is ever left half-written behind a later step.

set -euo pipefail

CK=ckpt_fix                       # checkpoints, alla radice del repo
OUT=risultati/fix                 # output, alla radice del repo
EVAL=src/chess_eval/evaluation    # gli script di valutazione
GRAPHS=src/chess_eval/graphs      # i grafi parquet
DATA=dataset                      # i CSV
mkdir -p "$OUT"

# ---------------------------------------------------------------- checks
echo "=== controlli prima di partire ==="
missing=0
for f in "$CK/timing/best.pt" "$CK/plain/best.pt" \
         "$EVAL/eval_ranks.py" "$EVAL/evaluate_models.py" "$EVAL/playout_models.py" \
         "$DATA/test.csv" "$DATA/test_clean.csv"; do
    if [ -e "$f" ]; then echo "  ok      $f"; else echo "  MANCA   $f"; missing=1; fi
done
for d in "$GRAPHS/test" "$GRAPHS/test_plain" "$GRAPHS/test_clean" "$GRAPHS/test_plain_clean"; do
    n=$(ls "$d"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -gt 0 ]; then echo "  ok      $d ($n parquet)"; else echo "  MANCA   $d"; missing=1; fi
done
[ "$missing" -eq 0 ] || { echo; echo "manca qualcosa: sistemalo e rilancia"; exit 1; }

python - <<PY
import torch
for tag in ("timing", "plain"):
    a = torch.load(f"ckpt_fix/{tag}/best.pt", map_location="cpu",
                   weights_only=False)["args"]
    ok = a.get("edge_inputs") and a.get("self_loops") \
         and a["timing"] == (tag == "timing")
    print(f"  {'ok' if ok else 'ERRORE':<7} {tag}: timing={a['timing']} "
          f"edge_inputs={a.get('edge_inputs')} self_loops={a.get('self_loops')}")
    assert ok, f"il checkpoint {tag} non e' quello giusto"
PY

# ---------------------------------------------------------------- runs
run() {   # run <name> <command...>: prints, logs, stops on failure
    local name=$1; shift
    echo
    echo "=================================================================="
    echo "=== $name   ($(date +%H:%M))"
    echo "=================================================================="
    "$@" 2>&1 | tee "$OUT/log_$name.txt"
}

for set in full clean; do
    if [ "$set" = full ]; then
        TD=$GRAPHS/test;       PD=$GRAPHS/test_plain;       CSV=$DATA/test.csv;       S=""
    else
        TD=$GRAPHS/test_clean; PD=$GRAPHS/test_plain_clean; CSV=$DATA/test_clean.csv; S="_clean"
    fi

    run "test$S" python "$EVAL/evaluate_models.py" \
        --timing-ckpt "$CK/timing/best.pt" --timing-data "$TD" \
        --plain-ckpt  "$CK/plain/best.pt"  --plain-data  "$PD" \
        --out "$OUT/results_test$S.json"

    run "ranks$S" python "$EVAL/eval_ranks.py" \
        --timing-ckpt "$CK/timing/best.pt" --timing-data "$TD" \
        --plain-ckpt  "$CK/plain/best.pt"  --plain-data  "$PD" \
        --topk 1 3 5 --out "$OUT/results_ranks$S.json"

    run "playout$S" python "$EVAL/playout_models.py" \
        --timing-ckpt "$CK/timing/best.pt" \
        --plain-ckpt  "$CK/plain/best.pt" \
        --test-csv "$CSV" --out "$OUT/results_playout$S.json"
done

echo
echo "=== finito ($(date +%H:%M)). Risultati in $OUT/:"
ls -1 "$OUT"