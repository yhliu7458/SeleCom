import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import sys
sys.path.append('..')
import torch
import multiprocessing as mp
from transformers import AutoTokenizer
from tqdm import tqdm


INPUT_FILE = '../../data/raw/corpus.pt'
OUTPUT_FILE = '../../data/raw/corpus_filtered_stage0.pt'
MODEL_NAME = 'path/to/your/model' # To be filled. We use Qwen3-Embedding-0.6B.
BATCH_SIZE = 1048576


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    with open(INPUT_FILE, 'rb') as f:
        all_docs = torch.load(f, weights_only=False)
    print(f'Before filtering, {len(all_docs)} data in total')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left', use_fast=True, trust_remote_code=True)

    final_data = []
    for start in tqdm(range(0, len(all_docs), BATCH_SIZE)):
        batch = all_docs[start:start + BATCH_SIZE]

        documents = tokenizer(
            batch,
            add_special_tokens=False,
            return_length=True
        )
        
        for item, length in zip(batch, documents["length"]):
            if length <= 512:
                final_data.append(item)
            
    print(f'After filtering, {len(final_data)} data left')

    torch.save(final_data, OUTPUT_FILE)
    print(f'Saved filtered data to {OUTPUT_FILE}')