#!/usr/bin/env python3
"""Patch LiveCodeBench for datasets>=4.0 and vLLM GPU/memory/tokenizer fixes."""
from __future__ import annotations

import sys
from pathlib import Path

DATASET_OLD = (
    'load_dataset("livecodebench/code_generation_lite", split="test", '
    "version_tag=release_version, trust_remote_code=True)"
)
DATASET_NEW = """load_dataset(
        "livecodebench/code_generation_lite",
        release_version,
        split="test",
        revision="refs/pr/7",
    )"""

VLLM_TOKENIZER_MARKER = "tokenizer_path = ("
VLLM_TOKENIZER_OLD = """        model_tokenizer_path = (
            model.model_name if args.local_model_path is None else args.local_model_path
        )
        self.llm = LLM(
            model=model_tokenizer_path,
            tokenizer=model_tokenizer_path,"""
VLLM_TOKENIZER_NEW = """        model_tokenizer_path = (
            model.model_name if args.local_model_path is None else args.local_model_path
        )
        weights_path = model_tokenizer_path
        tokenizer_path = (
            os.environ.get("TOKENIZER_PATH")
            or (model.model_name if args.local_model_path else model_tokenizer_path)
        )
        self.llm = LLM(
            model=weights_path,
            tokenizer=tokenizer_path,"""

VLLM_GPU_MARKER = 'gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION"'
VLLM_GPU_OLD = """        self.llm = LLM(
            model=weights_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            enforce_eager=True,"""
VLLM_GPU_NEW = """        self.llm = LLM(
            model=weights_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.55")),
            enforce_eager=True,"""


def patch_dataset(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if DATASET_NEW.strip() in text:
        return False
    if DATASET_OLD not in text:
        raise SystemExit(f"expected load_dataset call not found in {path}")
    path.write_text(text.replace(DATASET_OLD, DATASET_NEW), encoding="utf-8")
    return True


def patch_vllm_runner(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False
    if VLLM_TOKENIZER_MARKER not in text:
        if VLLM_TOKENIZER_OLD not in text:
            raise SystemExit(f"expected vllm_runner LLM() block not found in {path}")
        if "import os\n" not in text[:300]:
            text = text.replace("try:\n", "import os\n\ntry:\n", 1)
        text = text.replace(VLLM_TOKENIZER_OLD, VLLM_TOKENIZER_NEW, 1)
        changed = True
    if VLLM_GPU_MARKER not in text:
        old = VLLM_GPU_OLD if "weights_path" in text else VLLM_GPU_OLD.replace("weights_path", "model_tokenizer_path").replace("tokenizer_path", "model_tokenizer_path")
        if old not in text:
            raise SystemExit(f"expected vllm_runner gpu patch point not found in {path}")
        if "import os\n" not in text[:300]:
            text = text.replace("try:\n", "import os\n\ntry:\n", 1)
        text = text.replace(old, VLLM_GPU_NEW, 1)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def main() -> None:
    lcb_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("third_party/LiveCodeBench")
    dataset_py = lcb_dir / "lcb_runner" / "benchmarks" / "code_generation.py"
    vllm_py = lcb_dir / "lcb_runner" / "runner" / "vllm_runner.py"
    if not dataset_py.is_file():
        raise SystemExit(f"LiveCodeBench not found: {dataset_py}")
    ds_changed = patch_dataset(dataset_py)
    print(f"{'patched' if ds_changed else 'already patched'}: {dataset_py}")
    if vllm_py.is_file():
        vllm_changed = patch_vllm_runner(vllm_py)
        print(f"{'patched' if vllm_changed else 'already patched'}: {vllm_py}")


if __name__ == "__main__":
    main()
