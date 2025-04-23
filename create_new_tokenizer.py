#!/usr/bin/env python3
import argparse
from utils.tokenizer_trainer import train_and_update_tokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Train and update a BERT tokenizer with custom vocabulary."
    )

    parser.add_argument(
        "--data_directory",
        type=str,
        default="./data/",
        help="Path to the directory containing raw text data (default: %(default)s)"
    )
    parser.add_argument(
        "--trained_tokenizer_directory",
        type=str,
        default="tokenizer/trained_bert",
        help="Directory where the intermediate trained tokenizer will be saved (default: %(default)s)"
    )
    parser.add_argument(
        "--final_directory",
        type=str,
        default="tokenizer/modified_bert",
        help="Directory where the final updated tokenizer will be written (default: %(default)s)"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=30000,
        help="Final vocabulary size for the tokenizer (default: %(default)s)"
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=700,
        help="Number of new tokens to add (default: %(default)s)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["replace_unused", "add_new"],
        default="replace_unused",
        help="How to integrate new tokens: replace unused ids or append (default: %(default)s)"
    )
    parser.add_argument(
        "--prioritize_scibert",
        action="store_true",
        default="False",
        help="Whether to prioritize SCIBERT tokens when updating (default: False)"
    )

    args = parser.parse_args()

    train_and_update_tokenizer(
        data_directory=args.data_directory,
        trained_tokenizer_directory=args.trained_tokenizer_directory,
        final_directory=args.final_directory,
        vocab_size=args.vocab_size,
        num_tokens=args.num_tokens,
        mode=args.mode,
        prioritize_scibert=args.prioritize_scibert
    )

if __name__ == "__main__":
    main()
