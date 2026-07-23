import argparse
import json
import os

from datasets import load_dataset
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out_dir", required=True, help="dir to save per-problem JSONs")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--dtype", default=os.environ.get("DTYPE", "bfloat16"))
    ap.add_argument("--tokenizer_path", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset("nuprl/MultiPL-E", f"humaneval-{args.lang}", split="test")
    print(f"loaded {len(ds)} problems for {args.lang}")

    tokenizer_path = args.tokenizer_path or args.model

    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_path,
        tensor_parallel_size=args.tp,
        dtype=args.dtype,
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.75")),
        enforce_eager=True,
        trust_remote_code=True,
    )

    stop_tokens = ds[0]["stop_tokens"]
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=stop_tokens,
        n=1,
    )
    outputs = llm.generate([ex["prompt"] for ex in ds], sp)

    for ex, out in zip(ds, outputs):
        rec = {
            "name": ex["name"],
            "language": ex["language"],
            "prompt": ex["prompt"],
            "tests": ex["tests"],
            "stop_tokens": ex["stop_tokens"],
            "completions": [out.outputs[0].text],
        }
        with open(os.path.join(args.out_dir, f"{ex['name']}.json"), "w") as f:
            json.dump(rec, f)

    print(f"saved {len(ds)} files to {args.out_dir}")

    del llm
    import gc
    gc.collect()


if __name__ == "__main__":
    main()
