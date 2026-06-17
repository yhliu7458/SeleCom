import argparse
import os
import random
import sys
import time

import faiss
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from util.llm_utils import compute_trainable_parameters
from util.util import load_jsonl, save_jsonl


DEFAULT_ENCODER_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_INPUT_PATH = "../data/raw/qa_train.jsonl"
DEFAULT_OUTPUT_PATH = "../data/stage2/qda_train.jsonl"
DEFAULT_TMP_DIR = "../data/raw/qda_tmp"
DEFAULT_CORPUS_PATH = "../data/raw/corpus.pt"
DEFAULT_INDEX_PATH = "../data/raw/faiss.index"

ctx = mp.get_context("fork")

_shared_corpus = None
_shared_index = None


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


class SharedRetriever(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_length = args.encoder_max_length

        global _shared_corpus, _shared_index
        self.corpus = _shared_corpus
        self.the_index = _shared_index
        print(
            f"Process {os.getpid()}: using shared corpus "
            f"({len(self.corpus)} docs) and index ({self.the_index.ntotal} vectors)"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.encoder_name,
            padding_side="left",
            use_fast=True,
            trust_remote_code=True,
        )
        self.encoder = AutoModel.from_pretrained(
            args.encoder_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    def last_token_pooling(self, last_hidden_states):
        return last_hidden_states[:, -1]

    def retrieve(self, questions, k):
        questions = [
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            f"Query: {query}"
            for query in questions
        ]
        encoder_input = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoder_input_ids = encoder_input["input_ids"].to(self.encoder.device)
        encoder_attention_mask = encoder_input["attention_mask"].to(self.encoder.device)
        model_output = self.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
        )
        query_embedding = self.last_token_pooling(model_output.last_hidden_state)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)
        query_embedding = query_embedding.cpu().numpy().astype("float32")
        _, all_indices = self.the_index.search(query_embedding, k)

        results = []
        for indices in all_indices:
            results.append([self.corpus[index] for index in indices])
        return results


class SharedRetrieverModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = SharedRetriever(args)
        encoder_size = self.encoder.encoder.get_input_embeddings().weight.shape[-1]
        self.config = self.encoder.encoder.config
        print(f"Encoder dimension: {encoder_size}")

        self.set_trainable()
        compute_trainable_parameters(self)

    def set_trainable(self):
        for p in self.encoder.encoder.parameters():
            p.requires_grad = False
        print("Frozen encoder")

    def forward(self, questions, k):
        with torch.no_grad():
            return self.encoder.retrieve(questions, k)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attach retrieved documents to QA data using a prepared retrieval source."
    )
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--encoder_name", type=str, default=DEFAULT_ENCODER_NAME)
    parser.add_argument("--encoder_max_length", type=int, default=2048)
    parser.add_argument("--retrieve_top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tmp_dir", type=str, default=DEFAULT_TMP_DIR)
    parser.add_argument("--retrieval_source_corpus", type=str, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--retrieval_source_index", type=str, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def load_shared_resources(args):
    global _shared_corpus, _shared_index
    _shared_corpus = torch.load(args.retrieval_source_corpus, weights_only=False)
    print(f"Loaded corpus with {len(_shared_corpus)} documents")
    _shared_index = faiss.read_index(args.retrieval_source_index)
    print(f"Loaded index with {_shared_index.ntotal} vectors")


def validate_runtime(args):
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if args.world_size > device_count:
            raise ValueError(
                f"world_size={args.world_size} exceeds available CUDA devices ({device_count})."
            )
    elif args.world_size != 1:
        print("CUDA is not available; forcing world_size=1.")
        args.world_size = 1


def inference_worker(rank, args, data, counter, lock):
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    print(f"Process {rank} (PID: {os.getpid()}): initializing on {device}")

    model = SharedRetrieverModel(args).to(device)
    model.eval()

    total = len(data)
    per_worker = (total + args.world_size - 1) // args.world_size
    start = rank * per_worker
    end = min(total, start + per_worker)
    subset = data[start:end]

    results = []
    with torch.no_grad():
        for i in range(0, len(subset), args.batch_size):
            batch = subset[i : i + args.batch_size]
            questions = [item["question"] for item in batch]
            docs = model(questions, k=args.retrieve_top_k)
            for question, retrieved_docs, original in zip(questions, docs, batch):
                output_item = dict(original)
                output_item["question"] = question
                output_item["documents"] = original.get("documents", retrieved_docs)
                results.append(output_item)
            with lock:
                counter.value += len(batch)

    os.makedirs(args.tmp_dir, exist_ok=True)
    torch.save(results, os.path.join(args.tmp_dir, f"part_{rank}.pt"))
    print(f"Process {rank}: completed {len(results)} records")


def merge_results(tmp_dir, world_size):
    all_results = []
    for rank in range(world_size):
        part_path = os.path.join(tmp_dir, f"part_{rank}.pt")
        if not os.path.exists(part_path):
            continue
        part = torch.load(part_path, weights_only=False)
        all_results.extend(part)
    print(f"Merged {len(all_results)} records")

    for rank in range(world_size):
        part_path = os.path.join(tmp_dir, f"part_{rank}.pt")
        try:
            os.remove(part_path)
        except OSError:
            pass
    return all_results


def main():
    args = parse_args()
    validate_runtime(args)
    random.seed(args.seed)

    data = load_jsonl(args.input_path)
    data_with_documents = [item for item in data if "documents" in item]
    data_without_documents = [item for item in data if "documents" not in item]
    random.shuffle(data_without_documents)
    total = len(data_without_documents)

    load_shared_resources(args)

    manager = ctx.Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()

    processes = []
    for rank in range(args.world_size):
        p = ctx.Process(
            target=inference_worker,
            args=(rank, args, data_without_documents, counter, lock),
        )
        p.start()
        processes.append(p)

    pbar = tqdm(total=total, desc="Retrieval progress")
    previous = 0
    while any(p.is_alive() for p in processes):
        with lock:
            current = counter.value
        if current > previous:
            pbar.update(current - previous)
            previous = current
        time.sleep(0.1)

    with lock:
        current = counter.value
    if current > previous:
        pbar.update(current - previous)
    pbar.close()

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process {p.pid} exited with code {p.exitcode}.")

    all_results = merge_results(args.tmp_dir, args.world_size)
    all_results.extend(data_with_documents)
    random.shuffle(all_results)

    ensure_parent_dir(args.output_path)
    save_jsonl(all_results, args.output_path)
    print("Generated QDA data successfully.")


if __name__ == "__main__":
    main()
