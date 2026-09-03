import json
import re
import os
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai


PROBLEMS = [
    'r1bq3k/pppp1Bp1/2n4B/2b1p2Q/8/2P5/PPP2PPP/R4RK1 b - - 0 11',
    'rn2k2r/ppp2pp1/3b1n2/1N2q3/2Q1p2p/1PN1P2P/PB1P1PP1/R4RK1 w kq - 6 15',
    '3r2k1/pp3pp1/3r1qnp/3P4/1P2R3/P4B2/2Q3PP/4R2K b - - 3 30'
]

'''
sol:
    g7h6 h5h6
    a1c1 e5h2
    g6h4 e4e8 d8e8 e1e8
'''

PROMPT = 'Given this FEN position for a mate in N moves problem find the solution. Answer with the following format {FEN:<FEN of the solved problem board>, moves:<the list of moves that solve the problem, using standard notation>, solved: <True/False>}. If you ever end up in an illegal position stop reasoning and return the last valid board FEN and the moves sequence that brought you to it'


def init_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable.")
    client = genai.Client(api_key=api_key)
    return client


def query_llm(
    client: genai.Client,
    query: str,
    model: str = "gemini-3.6-flash"
):

    try:
        response = client.models.generate_content(
            model=model,
            contents=query
        )
    except genai.errors.ServerError as e:
        print(e)
        raise e
    return response.text


def parse_response_to_dict(response_text: str):
    if not response_text:
        return None
    try:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            return None
        parsed = json.loads(json_match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


if __name__ == '__main__':
    c = init_client()
    res = []
    for p in PROBLEMS:
        try:
            r = query_llm(c, p+' '+PROMPT)
            print(parse_response_to_dict(r))
        except genai.errors.ServerError as e:
            r = query_llm(c, p+' '+PROMPT)
            print(parse_response_to_dict(r))
