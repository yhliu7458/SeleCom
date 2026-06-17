# Retrieval Source

This folder contains the scripts used to build the retrieval source and attach retrieved documents to QA data.

The three scripts are connected by the same default paths:

1. `gen_corpus.py` converts source JSONL documents into `../data/raw/corpus.pt`.
2. `gen_retrieval_source.py` encodes `../data/raw/corpus.pt` and builds `../data/raw/faiss.index`.
3. `gen_qda_data.py` uses `../data/raw/corpus.pt` and `../data/raw/faiss.index` to generate QDA data with retrieved documents.

## Usage

```bash
python gen_corpus.py \
  --input_files ../data/raw/wiki18_corpus.jsonl \
  --output_path ../data/raw/corpus.pt
```

Build embeddings and the FAISS index:

```bash
python gen_retrieval_source.py \
  --model_name Qwen/Qwen3-Embedding-0.6B \
  --corpus_path ../data/raw/corpus.pt \
  --index_path ../data/raw/faiss.index
```

Generate QDA data:

```bash
python gen_qda_data.py \
  --input_path ../data/raw/qa_train.jsonl \
  --output_path ../data/stage2/qda_train.jsonl \
  --retrieval_source_corpus ../data/raw/corpus.pt \
  --retrieval_source_index ../data/raw/faiss.index
```

The source document JSONL defaults to `title` and `text` fields. The QA JSONL should contain `question` and `answer`; existing `documents` fields are preserved.

Large generated files such as `corpus.pt`, `corpus_embeddings.npy`, and `faiss.index` are not stored in this Git repository.
