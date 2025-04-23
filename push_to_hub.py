#!/usr/bin/env python
import argparse
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Push a locally saved Hugging Face tokenizer to the Hub"
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="./tokenizer/modified_bert",
        help="Path to the local tokenizer folder (default: ./tokenizer/modified_bert)",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default="BASF-AI/modified_bert",
        help="The Hub repository name, e.g. username/model-name (default: BASF-AI/modified_bert)",
    )
    parser.add_argument(
        "--auth_token",
        type=str,
        default=None,
        help="Your Hugging Face auth token (or set HF_HOME/token via CLI)",
    )

    args = parser.parse_args()

    # Load your tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # Push to Hub
    push_kwargs = {}
    if args.auth_token:
        push_kwargs["use_auth_token"] = args.auth_token

    tokenizer.push_to_hub(args.repo_name, **push_kwargs)
    print(f"✅ Tokenizer at '{args.tokenizer_dir}' pushed to 'https://huggingface.co/{args.repo_name}'")

if __name__ == "__main__":
    main()