import argparse
from huggingface_hub import HfApi


IGNORE_PATTERNS = [
    "optimizer.pt",
    "optimizer.bin",
    "scheduler.pt",
    "rng_state*.pth",
    "rng_state*.pt",
    "trainer_state.json",
    "training_args.bin",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="local path to the trained checkpoint dir")
    ap.add_argument("--eval_dir", default=None, help="optional path to eval results dir (uploaded as /eval inside the repo)")
    ap.add_argument("--repo_id", required=True, help="target hf repo id, e.g. user/qwen2.5-7b-tooluse-rho0.5")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--commit_message", default="upload checkpoint + eval")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, exist_ok=True, private=args.private)

    print(f"uploading checkpoint from {args.checkpoint} to {args.repo_id}")
    api.upload_folder(
        folder_path=args.checkpoint,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        ignore_patterns=IGNORE_PATTERNS,
    )

    if args.eval_dir:
        print(f"uploading eval results from {args.eval_dir} to {args.repo_id}/eval")
        api.upload_folder(
            folder_path=args.eval_dir,
            repo_id=args.repo_id,
            path_in_repo="eval",
            commit_message="upload eval results",
        )

    print(f"uploaded: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
