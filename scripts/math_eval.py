import argparse
import json
import os
from math import comb

import numpy as np
import torch
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

from math_grader import boxed_reward_fn, extract_answer

PASS_TEMPERATURE = 0.7
ACC_TEMPERATURE = 0.0
SYSTEM_PROMPT = "Reason step by step, and put your final answer within \\boxed{}."


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _normalize_dataset(dataset: Dataset) -> Dataset:
    """Normalize to {problem, answer}."""
    cols = dataset.column_names

    def row_fn(ex):
        if "problem" in ex:
            problem = ex["problem"]
        elif "Problem" in ex:
            problem = ex["Problem"]
        elif "question" in ex:
            problem = ex["question"]
        else:
            raise KeyError(f"no problem column in {cols}")

        if "answer" in ex:
            answer = ex["answer"]
        elif "Answer" in ex:
            answer = ex["Answer"]
        else:
            raise KeyError(f"no answer column in {cols}")
        return {"problem": problem, "answer": str(answer)}

    return dataset.map(row_fn, remove_columns=cols)


def load_math_dataset(name: str) -> tuple[Dataset, str]:
    if name == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        subdir = "aime24"
    elif name == "aime25":
        ds = load_dataset("math-ai/aime25", split="test")
        subdir = "aime25"
    elif name == "amc":
        ds = load_dataset("math-ai/amc23", split="test")
        subdir = "amc23"
    elif name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        subdir = "math500"
    elif name == "hmmt":
        ds = load_dataset("MathArena/hmmt_feb_2024", split="train")
        subdir = "hmmt"
    elif name == "minerva":
        ds = load_dataset("math-ai/minervamath", split="test")
        subdir = "minerva"
    elif name == "beyondaime":
        ds = load_dataset("ByteDance-Seed/BeyondAIME", split="test")
        subdir = "beyondaime"
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return _normalize_dataset(ds), subdir


def _max_gen_toks_from_gen_kwargs(gen_kwargs: str) -> int:
    for part in gen_kwargs.split(","):
        part = part.strip()
        if part.startswith("max_gen_toks="):
            return int(part.split("=", 1)[1])
    return 2048


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["aime", "aime25", "amc", "math500", "hmmt", "minerva", "beyondaime"],
    )
    p.add_argument("--base_folder", type=str, default=None,
                   help="defaults to <cwd>/benchmark/math_passk")
    p.add_argument("--gen_kwargs", type=str,
                   default=os.environ.get("GEN_KWARGS", "max_gen_toks=2048"))
    p.add_argument("--max_k", type=int, default=16,
                   help="samples per problem for pass@k (default: 16)")
    p.add_argument("--tp", type=int, default=None,
                   help="tensor parallel size (default: all visible GPUs)")
    p.add_argument("--gpu_memory_utilization", type=float,
                   default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.75")))
    p.add_argument("--dtype", type=str, default=os.environ.get("DTYPE", "bfloat16"))
    p.add_argument("--tokenizer_path", type=str, default=None,
                   help="tokenizer HF path (auto-detected if broken)")
    args = p.parse_args()

    max_k = args.max_k
    max_tokens = _max_gen_toks_from_gen_kwargs(args.gen_kwargs)
    ks = [k for k in [1, 2, 4, 8, 16, 32, 64] if k <= max_k]

    dataset, subdir = load_math_dataset(args.dataset)
    base_dir = args.base_folder or os.path.join(os.getcwd(), "benchmark", "math_passk", subdir)

    tp = args.tp or max(1, torch.cuda.device_count())
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=tp,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    prompts = []
    for item in dataset:
        if "llama" in args.model_path.lower():
            prompts.append(
                f"{SYSTEM_PROMPT}\n\nProblem:\n{item['problem']}\n\nSolution:\n"
            )
        else:
            prompts.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": item["problem"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

    pass_sampling = SamplingParams(
        temperature=PASS_TEMPERATURE, max_tokens=max_tokens, n=max_k,
    )
    acc_sampling = SamplingParams(
        temperature=ACC_TEMPERATURE, max_tokens=max_tokens, n=1,
    )

    pass_outputs = llm.generate(prompts, pass_sampling)
    acc_outputs = llm.generate(prompts, acc_sampling)

    data_path = os.path.join(base_dir, f"{args.exp_name}.jsonl")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if os.path.exists(data_path):
        os.remove(data_path)

    all_n_correct = []
    acc_corrects = []

    for pass_output, acc_output, item in zip(pass_outputs, acc_outputs, dataset):
        item = dict(item)
        gold = item["answer"]

        pass_solutions = [comp.text for comp in pass_output.outputs]
        pass_answers = []
        pass_corrects = []
        for sol in pass_solutions:
            _, reward = boxed_reward_fn(sol, gold)
            pass_answers.append(extract_answer(sol))
            pass_corrects.append(bool(reward > 0))

        acc_solution = acc_output.outputs[0].text
        acc_answer = extract_answer(acc_solution)
        _, acc_reward = boxed_reward_fn(acc_solution, gold)
        acc_correct = bool(acc_reward > 0)

        item.update(
            {
                "pass_predicted_answers": pass_answers,
                "pass_correct": pass_corrects,
                "pass_solutions": pass_solutions,
                "acc_predicted_answer": acc_answer,
                "acc_correct": acc_correct,
                "acc_solution": acc_solution,
            }
        )

        with open(data_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        all_n_correct.append(sum(pass_corrects))
        acc_corrects.append(acc_correct)

    n_problems = len(dataset)
    summary = {
        "model": args.exp_name,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "pass_temperature": PASS_TEMPERATURE,
        "acc_temperature": ACC_TEMPERATURE,
        "n": max_k,
    }
    for k in ks:
        vals = [pass_at_k(max_k, c, k) for c in all_n_correct]
        summary[f"pass@{k}"] = float(np.mean(vals))

    summary[f"acc@{max_k}"] = float(np.mean(all_n_correct) / max_k)
    summary["acc@1_greedy"] = float(np.mean(acc_corrects))
    summary["n_correct_acc@1_greedy"] = int(sum(acc_corrects))
    summary["n_total"] = n_problems

    summary_path = os.path.join(base_dir, "summary.jsonl")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    out_json = os.path.join(base_dir, f"{args.exp_name}_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"{args.exp_name} - {args.dataset}")
    print(f"pass@k: n={max_k}, T={PASS_TEMPERATURE}")
    print(f"acc@1_greedy: n=1, T={ACC_TEMPERATURE}")
    for k in ks:
        print(f"pass@{k}: {summary[f'pass@{k}']:.4f}")
    print(f"acc@{max_k}: {summary[f'acc@{max_k}']:.4f}")
    print(
        f"acc@1_greedy: {summary['acc@1_greedy']:.4f} "
        f"({summary['n_correct_acc@1_greedy']}/{n_problems})"
    )
    print(f"saved: {out_json}")

    del llm
    import gc
    gc.collect()


if __name__ == "__main__":
    main()
