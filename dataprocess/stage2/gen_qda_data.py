import sys
sys.path.append('../..')

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import argparse
import time
import torch.multiprocessing as mp
import faiss
from util.util import load_jsonl, save_jsonl
from util.llm_utils import compute_trainable_parameters
from transformers import AutoModel, AutoTokenizer

ctx = mp.get_context('fork')  


_shared_corpus = None
_shared_index = None

class SharedRetriever(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_length = args.encoder_max_length
        global _shared_corpus, _shared_index
        
        self.corpus = _shared_corpus
        self.the_index = _shared_index
        print(f'Process {os.getpid()}: Using shared corpus ({len(self.corpus)} docs) and index ({self.the_index.ntotal} vectors)')

        self.tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, padding_side='left', use_fast=True, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(args.encoder_name, torch_dtype=torch.float32)

    def last_token_pooling(self, last_hidden_states):
        return last_hidden_states[:, -1]
        
    def retrieve(self, questions, k):
        questions = [f'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}' for query in questions]
        encoder_input = self.tokenizer(
            questions, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        encoder_input_ids = encoder_input['input_ids'].to(self.encoder.device)
        encoder_attention_mask = encoder_input['attention_mask'].to(self.encoder.device)
        model_output = self.encoder(
            input_ids=encoder_input_ids, 
            attention_mask=encoder_attention_mask
        )
        query_embedding = self.last_token_pooling(model_output.last_hidden_state)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=-1).cpu().numpy()
        _, all_indices = self.the_index.search(query_embedding, k)
        results = []
        for indices in all_indices:
            individual_result = []
            for index in indices:
                individual_result.append(self.corpus[index])
            results.append(individual_result)
        return results

class SharedRetrieverModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = SharedRetriever(args)
        encoder_size = self.encoder.encoder.embed_tokens.weight.shape[-1]
        self.config = self.encoder.encoder.config
        print(f'Encoder dimension: {encoder_size}')
        
        self.set_trainable()
        compute_trainable_parameters(self)
        
    def set_trainable(self):
        for p in self.encoder.encoder.parameters():
            p.requires_grad = False
        print(f'Frozen encoder')

    def forward(self, questions, k):
        with torch.no_grad():
            return self.encoder.retrieve(questions, k)

def load_shared_resources(args):
    global _shared_corpus, _shared_index
    _shared_corpus = torch.load(args.retrieval_source_corpus, weights_only=False)
    print(f'Loaded corpus with {len(_shared_corpus)} documents')
    _shared_index = faiss.read_index(args.retrieval_source_index)
    print(f'Loaded index with {_shared_index.ntotal} vectors')

def inference_worker(rank, args, datas, output_dir, counter, lock):
    torch.cuda.set_device(rank)
    args.device = f"cuda:{rank}"
    
    print(f"Process {rank} (PID: {os.getpid()}): Initializing with shared resources")
    
    model = SharedRetrieverModel(args).to(args.device)
    model.eval()

    N = len(datas)
    per = (N + args.world_size - 1) // args.world_size
    start = rank * per
    end = min(N, start + per)
    subset = datas[start:end]

    results = []
    with torch.no_grad():
        for i in range(0, len(subset), args.batch_size):
            batch = subset[i : i + args.batch_size]
            questions = [d['question'] for d in batch]
            docs = model(questions, k=args.retrieve_top_k)
            for q, d, orig in zip(questions, docs, batch):
                results.append({
                    'question':   q,
                    'documents':  orig.get('documents', d),
                    'answer':     orig['answer']
                })
            # bump the shared counter
            with lock:
                counter.value += len(batch)

    torch.save(results, os.path.join(output_dir, f"part_{rank}.pt"))
    print(f"Process {rank}: Completed processing")

def merge_results(output_dir, world_size):
    all_results = []
    for rank in range(world_size):
        part = torch.load(os.path.join(output_dir, f"part_{rank}.pt"))
        all_results.extend(part)
    print(f"Merged {world_size} parts into ({len(all_results)} records)")
    for rank in range(world_size):
        try:
            os.remove(os.path.join(output_dir, f"part_{rank}.pt"))
        except OSError:
            pass
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='../../data/qa/qa_train_new.jsonl')
    parser.add_argument("--encoder_name", type=str, default='path/to/your/model') # To be filled. We use Qwen3-embedding-0.6B.
    parser.add_argument("--encoder_max_length", type=int, default=2048)
    parser.add_argument("--retrieve_top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--output_path", type=str, default='../../data/qa/qda_train_new.jsonl')
    parser.add_argument("--tmp_dir", type=str, default='../../data/raw')
    parser.add_argument("--retrieval_source_corpus", type=str, default='../../data/raw/corpus.pt')
    parser.add_argument("--retrieval_source_index", type=str, default='../../data/raw/faiss.index')
    parser.add_argument("--world_size", type=int, default=4)
    args = parser.parse_args()

    data = load_jsonl(args.input_path)
    has_document_data = [item for item in data if 'documents' in item]
    non_document_data = [item for item in data if 'documents' not in item]
    random.shuffle(non_document_data)
    total = len(non_document_data)

    load_shared_resources(args)

    # use Manager only for progress tracking
    manager = ctx.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    processes = []
    for rank in range(args.world_size):
        p = ctx.Process(
            target=inference_worker,
            args=(rank, args, non_document_data, args.tmp_dir, counter, lock)
        )
        p.start()
        processes.append(p)

    pbar = tqdm(total=total, desc="Overall progress")
    prev = 0
    while any(p.is_alive() for p in processes):
        with lock:
            curr = counter.value
        delta = curr - prev
        if delta > 0:
            pbar.update(delta)
            prev = curr
        time.sleep(0.1)
    # final catch-up
    with lock:
        curr = counter.value
    if curr > prev:
        pbar.update(curr - prev)
    pbar.close()

    for p in processes:
        p.join()

    all_results = merge_results(args.tmp_dir, args.world_size)
    all_results.extend(has_document_data)
    random.shuffle(all_results)
    print("All processes completed successfully!")
    save_jsonl(all_results, args.output_path)


if __name__ == "__main__":
    main()
