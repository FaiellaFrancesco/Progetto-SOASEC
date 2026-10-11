# Run (find which part of a request an API refuses):
#   python src/chess_eval/evaluation_llm/sonda_api.py --model gemma-4-31b-it \
#       --heldout dataset/heldout.csv --shots-file dataset/fewshot.json
#
# Sends the same question in progressively richer forms and prints each HTTP
# code, so the first form that fails names the culprit. Reads the key from
# GEMINI_API_KEY and never prints it.

import argparse
import os
import time
from types import SimpleNamespace

import requests

import chess_eval.evaluation_llm.valuta_heldout as v

URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def send(model, messages, max_tokens, key):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 1.0}
    t0 = time.time()
    r = requests.post(URL, json=body, timeout=600,
                      headers={"Authorization": f"Bearer {key}"})
    took = time.time() - t0
    try:
        j = r.json()
        if isinstance(j, list):
            j = j[0]
        usage = j.get("usage", {})
        text = (j.get("choices") or [{}])[0].get("message", {}).get("content")
        info = (f"token {usage.get('total_tokens')}  risposta: "
                f"{(text or '')[-80:]!r}" if r.ok else
                j.get("error", {}).get("message", r.text[:200]))
    except ValueError:
        info = r.text[:200]
    return r.status_code, took, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--shots-file", required=True)
    ap.add_argument("--id", default="harvey_n2_0020")
    ap.add_argument("--pezzi", action="store_true",
                    help="spezza la domanda nei suoi blocchi e li aggiunge "
                         "uno alla volta: il primo che fa fallire e' il colpevole")
    ap.add_argument("--ripeti", type=int, nargs=2, metavar=("BLOCCHI", "VOLTE"),
                    help="manda i blocchi 1..BLOCCHI per VOLTE volte: dice se un "
                         "errore e' sempre uguale (contenuto) o a caso (servizio)")
    a = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("manca GEMINI_API_KEY in questo terminale")

    g = v.load_games(a.heldout, ids=[a.id])[0]
    plain = SimpleNamespace(board=True, legal=True, system_in_user=True)
    examples = v.load_examples(a.shots_file, a.heldout)
    question = v.build_prompt(g, plain, "move")              # no shots
    with_shots = v.build_prompt(g, plain, "move",
                                v.fewshot_messages(examples, plain, "move"))
    words = sum(len(m["content"].split()) for m in with_shots)

    if a.ripeti:
        n, times = a.ripeti
        blocks = question[-1]["content"].split("\n\n")
        text = "\n\n".join(blocks[:n])
        ok = 0
        for k in range(1, times + 1):
            code, took, info = send(a.model, [{"role": "user", "content": text}],
                                    50, key)
            ok += code == 200
            print(f"  tentativo {k}: HTTP {code}  {took:4.1f} s")
            time.sleep(16)
        print(f"\n{ok} su {times} riusciti con i blocchi 1-{n}: "
              + ("errore sempre uguale -> dipende dal CONTENUTO" if ok == 0
                 else "errore a caso -> e' il SERVIZIO di Google"))
        return

    if a.pezzi:
        # the question is made of blocks joined by a blank line (build_prompt
        # joins them that way): grow it one block at a time
        blocks = question[-1]["content"].split("\n\n")
        for i in range(1, len(blocks) + 1):
            text = "\n\n".join(blocks[:i])
            code, took, info = send(a.model, [{"role": "user", "content": text}],
                                    50, key)
            head = blocks[i - 1].replace("\n", " ")[:50]
            print(f"blocchi 1-{i:<2} (+ {head!r:<54}) HTTP {code}  {took:4.1f} s")
            if code != 200:
                print(f"\n>>> il blocco {i} fa fallire la richiesta. Da solo:")
                code, took, info = send(a.model, [{"role": "user",
                                                   "content": blocks[i - 1]}],
                                        50, key)
                print(f"    solo blocco {i}: HTTP {code}\n--- testo del blocco ---\n"
                      f"{blocks[i - 1]}")
                return
            time.sleep(16)
        print("\nnessun blocco fa fallire la richiesta da solo")
        return

    variants = [
        ("1 domanda corta, 50 token", [{"role": "user", "content": "Say hi"}], 50),
        ("2 domanda vera, senza esempi, 50 token", question, 50),
        ("3 domanda vera, senza esempi, 8192 token", question, 8192),
        (f"4 con i 3 esempi (~{words} parole), 8192 token", with_shots, 8192),
    ]
    for name, msgs, mt in variants:
        roles = "".join(m["role"][0] for m in msgs)
        code, took, info = send(a.model, msgs, mt, key)
        print(f"{name:<45} ruoli={roles:<8} HTTP {code}  {took:5.1f} s  {info}")
        time.sleep(16)          # stays under 4 requests per minute


if __name__ == "__main__":
    main()