import argparse
import os

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_CORPUS_PATH = "../data/raw/corpus.pt"
DEFAULT_EMBEDDING_PATH = "../data/raw/corpus_embeddings.npy"
DEFAULT_INDEX_PATH = "../data/raw/faiss.index"


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode the corpus and build the FAISS retrieval source."
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--corpus_path", type=str, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--embedding_path", type=str, default=DEFAULT_EMBEDDING_PATH)
    parser.add_argument("--index_path", type=str, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--batch_size", type=int, default=800)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hnsw_m", type=int, default=32)
    parser.add_argument("--ef_construction", type=int, default=64)
    parser.add_argument("--ef_search", type=int, default=32)
    parser.add_argument(
        "--multi_process",
        action="store_true",
        help="Use sentence-transformers multi-process encoding.",
    )
    parser.add_argument(
        "--no_save_embeddings",
        action="store_true",
        help="Build the FAISS index without keeping a .npy copy of embeddings.",
    )
    return parser.parse_args()


def encode_corpus(model, corpus, args):
    if args.multi_process:
        pool = model.start_multi_process_pool()
        try:
            embeddings = model.encode_multi_process(
                sentences=corpus,
                pool=pool,
                batch_size=args.batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        finally:
            model.stop_multi_process_pool(pool)
        return embeddings

    return model.encode(
        corpus,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def build_index(embeddings, args):
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexHNSWFlat(embeddings.shape[1], args.hnsw_m)
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    index.hnsw.efConstruction = args.ef_construction
    index.hnsw.efSearch = args.ef_search
    index.add(embeddings)
    return index, embeddings


def main():
    args = parse_args()

    corpus = torch.load(args.corpus_path, weights_only=False)
    print(f"Loaded corpus from {args.corpus_path}: {len(corpus)} documents")

    model = SentenceTransformer(
        args.model_name,
        device=args.device,
        trust_remote_code=True,
        tokenizer_kwargs={
            "max_length": args.max_length,
            "truncation": True,
            "padding_side": "left",
        },
    )
    model.max_seq_length = args.max_length
    print(f"Loaded embedding model: {args.model_name}")

    embeddings = encode_corpus(model, corpus, args)
    print("Generated corpus embeddings")

    index, embeddings = build_index(embeddings, args)

    ensure_parent_dir(args.index_path)
    faiss.write_index(index, args.index_path)
    print(f"Saved FAISS index to {args.index_path}")

    if not args.no_save_embeddings:
        ensure_parent_dir(args.embedding_path)
        np.save(args.embedding_path, embeddings)
        print(f"Saved embeddings to {args.embedding_path}")


if __name__ == "__main__":
    main()
