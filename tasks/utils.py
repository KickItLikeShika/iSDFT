from math import comb
from typing import Dict, List

from lm_eval.tasks.aime.utils import (
    is_equiv,
    last_boxed_only_string,
    remove_boxed,
)


def _score_response(doc: dict, response: str) -> int:
    indices = [pos for pos, char in enumerate(response) if char == "$"]
    if len(indices) <= 1:
        answer = response
    else:
        answer = response[indices[0] + 1 : indices[-1]]

    boxed_answer = last_boxed_only_string(response)
    if boxed_answer is not None:
        try:
            boxed_content = remove_boxed(boxed_answer)
            if boxed_content is not None:
                answer = boxed_content
        except (AssertionError, IndexError):
            pass

    answer_key = next(k for k in doc.keys() if k.lower() == "answer")
    target = str(doc[answer_key])
    return 1 if is_equiv(answer, target) else 0


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def process_results(doc: dict, results: List[str]) -> Dict[str, float]:
    return {"exact_match": _score_response(doc, results[0])}


def process_results_passk(doc: dict, results: List[str]) -> Dict[str, float]:
    # set repeats: 16, do_sample: true in the yaml to use this
    scores = [_score_response(doc, r) for r in results]
    n, c = len(scores), sum(scores)
    out: Dict[str, float] = {"acc@1_greedy_first": float(scores[0]) if scores else 0.0}
    for k in (1, 2, 4, 8, 16):
        if k <= n:
            out[f"pass@{k}"] = pass_at_k(n, c, k)
    return out
