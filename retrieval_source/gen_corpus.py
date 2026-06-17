import argparse
import json
import os

import torch
from tqdm import tqdm


DEFAULT_INPUT_FILES = ["../data/raw/wiki18_corpus.jsonl"]
DEFAULT_OUTPUT_PATH = "../data/raw/corpus.pt"


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert JSONL documents into the torch corpus used by the retrieval source."
    )
    parser.add_argument(
        "--input_files",
        nargs="+",
        default=DEFAULT_INPUT_FILES,
        help="One or more JSONL files containing source documents.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the serialized corpus list.",
    )
    parser.add_argument(
        "--title_field",
        type=str,
        default="title",
        help="JSON field used as the document title.",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="JSON field used as the document body.",
    )
    return parser.parse_args()


def format_document(item, title_field, text_field):
    title = item.get(title_field, "")
    text = item.get(text_field, "")
    if title:
        return f"Title: {title}\nContent: {text}"
    return str(text)


def main():
    args = parse_args()
    all_texts = []

    print("Start reading source documents...")
    for file_path in args.input_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=os.path.basename(file_path)):
                if not line.strip():
                    continue
                item = json.loads(line)
                all_texts.append(format_document(item, args.title_field, args.text_field))

    if not all_texts:
        raise ValueError("No documents were loaded from the input files.")

    print("#" * 40)
    print("Example document:")
    print(all_texts[-1])
    print("#" * 40)
    print(f"Loaded {len(all_texts)} documents.")

    ensure_parent_dir(args.output_path)
    torch.save(all_texts, args.output_path)
    print(f"Saved corpus to {args.output_path}")


if __name__ == "__main__":
    main()
