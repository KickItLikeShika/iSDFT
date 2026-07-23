#!/usr/bin/env python3
"""Summarize MultiPL-E pass@1 per language and averaged across languages."""
import argparse
import json
import glob
import os


def summarize(base: str) -> None:
    lang_stats = []
    total_ok = total_n = 0

    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(d):
            continue
        lang = os.path.basename(d).split("_")[-1]
        files = glob.glob(os.path.join(d, "*.results.json"))
        ok = sum(
            1
            for f in files
            if any(r.get("status") == "OK" for r in json.load(open(f)).get("results", []))
        )
        n = len(files)
        if n == 0:
            continue
        rate = 100.0 * ok / n
        lang_stats.append((lang, ok, n, rate))
        total_ok += ok
        total_n += n
        print(f"{lang}: {ok}/{n} = {rate:.1f}%")

    if not lang_stats:
        print("no results found")
        return

    macro = sum(r for _, _, _, r in lang_stats) / len(lang_stats)
    micro = 100.0 * total_ok / total_n
    print(f"macro_avg (mean of lang %): {macro:.1f}%")
    print(f"micro_avg (all problems):   {total_ok}/{total_n} = {micro:.1f}%")


def main():
    p = argparse.ArgumentParser(description="Summarize MultiPL-E pass@1")
    p.add_argument(
        "results_dir",
        nargs="?",
        default="benchmark_results/KickItLikeShika_qwen-2.5-7b-instruct-sdft-tooluse_math_coding/multipl_e/results",
    )
    args = p.parse_args()
    summarize(args.results_dir)


if __name__ == "__main__":
    main()
