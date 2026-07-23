#!/usr/bin/env python3
"""Patch evalplus VllmDecoder to honor VLLM_GPU_MEMORY_UTILIZATION."""
from __future__ import annotations

import os
import sys
from pathlib import Path

MARKER = "gpu_memory_utilization=float(os.environ.get"
MAX_TOK_MARKER = 'max_new_tokens=int(os.environ.get("EVALPLUS_MAX_NEW_TOKENS"'
SNIPPET = """        "gpu_memory_utilization": float(
            os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.65")
        ),"""


def patch_gpu_memory(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    gpu_block = (
        '"gpu_memory_utilization": float(\n'
        '                os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.65")\n'
        "            ),"
    )
    while text.count('"gpu_memory_utilization":') > 1:
        text = text.replace(gpu_block + "\n            " + gpu_block, gpu_block, 1)
    if '"gpu_memory_utilization":' in text:
        path.write_text(text, encoding="utf-8")
        return False
    needle = '"enable_prefix_caching": True,'
    if needle not in text:
        raise SystemExit(f"unexpected evalplus vllm.py layout: {path}")
    if "import os\n" not in text[:300]:
        text = text.replace("from typing import List\n", "import os\nfrom typing import List\n", 1)
    text = text.replace(
        needle,
        needle + "\n            " + gpu_block,
        1,
    )
    path.write_text(text, encoding="utf-8")
    return True


def patch_max_new_tokens(init_py: Path) -> bool:
    text = init_py.read_text(encoding="utf-8")
    # Repair double-patched files from older patch runs.
    text = text.replace(
        '        max_new_tokens = int(os.environ.get("EVALPLUS_MAX_NEW_TOKENS", "2048"))\n'
        '        max_new_tokens = int(os.environ.get("EVALPLUS_MAX_NEW_TOKENS", "2048"))\n',
        '        max_new_tokens = int(os.environ.get("EVALPLUS_MAX_NEW_TOKENS", "2048"))\n',
    )
    text = text.replace(
        "            max_new_tokens=max_new_tokens,\n"
        "            max_new_tokens=max_new_tokens,\n",
        "            max_new_tokens=max_new_tokens,\n",
    )
    if 'max_new_tokens=max_new_tokens,\n            batch_size=batch_size' in text:
        init_py.write_text(text, encoding="utf-8")
        return False
    if MAX_TOK_MARKER in text:
        init_py.write_text(text, encoding="utf-8")
        return False
    needle = "        return VllmDecoder(\n            name=model,"
    if needle not in text:
        raise SystemExit(f"unexpected evalplus provider/__init__.py layout: {init_py}")
    if "import os\n" not in text[:200]:
        text = text.replace(
            "from evalplus.provider.base import DecoderBase\n",
            "import os\n\nfrom evalplus.provider.base import DecoderBase\n",
            1,
        )
    insert = (
        "        max_new_tokens = int(os.environ.get(\"EVALPLUS_MAX_NEW_TOKENS\", \"2048\"))\n"
        "        return VllmDecoder(\n            name=model,\n"
        "            max_new_tokens=max_new_tokens,"
    )
    text = text.replace(
        "        return VllmDecoder(\n            name=model,",
        insert,
        1,
    )
    init_py.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    import importlib.util

    spec = importlib.util.find_spec("evalplus")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("evalplus not installed")
    pkg_root = Path(list(spec.submodule_search_locations)[0])
    target = pkg_root / "provider" / "vllm.py"
    if not target.is_file():
        raise SystemExit(f"evalplus vllm.py not found: {target}")
    changed = patch_gpu_memory(target)
    print(f"{'patched' if changed else 'already patched'}: {target}")
    init_py = pkg_root / "provider" / "__init__.py"
    if init_py.is_file():
        init_changed = patch_max_new_tokens(init_py)
        print(f"{'patched' if init_changed else 'already patched'}: {init_py}")


if __name__ == "__main__":
    main()
