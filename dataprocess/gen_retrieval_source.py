import sys
sys.path.append('..')
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = 'path/to/your/model' # To be filled. We use Qwen3-Embedding-0.6B.
BATCH_SIZE = 800
MAX_LENGTH = 256

INPUT_FILE = '../data/raw/corpus.pt'
OUTPUT_NPY = '../data/raw/embedding.npy'
OUTPUT_FAISS = '../data/raw/faiss.index'

def main():
    all_texts = torch.load(INPUT_FILE, weights_only=False)
    print('Loaded corpus')

    model = SentenceTransformer(MODEL_NAME, device='cpu', model_kwargs={'torch_dtype': torch.bfloat16, 'attn_implementation': 'flash_attention_2'}, tokenizer_kwargs={'max_length': MAX_LENGTH, 'truncation': True, 'padding_side': 'left'})
    model.max_seq_length=MAX_LENGTH
    print('Loaded model')

    pool = model.start_multi_process_pool()
    final_embeddings = model.encode_multi_process(
        sentences=all_texts,
        pool=pool,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    model.stop_multi_process_pool(pool)
    print('Generated embeddings')

    np.save(OUTPUT_NPY, final_embeddings)
    print("Saved embeddings")

    the_index = faiss.IndexHNSWFlat(final_embeddings.shape[1], 32)
    the_index.metric_type = faiss.METRIC_INNER_PRODUCT
    the_index.hnsw.efConstruction = 64
    the_index.hnsw.efSearch = 32     
    the_index.add(final_embeddings)
    faiss.write_index(the_index, OUTPUT_FAISS)
    print('Created faiss index....')

if __name__ == '__main__':
    main()
