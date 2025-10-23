import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import torch
import multiprocessing as mp
from typing import List
from tqdm import tqdm
import sys
sys.path.append('..')
import logging


INPUT_FILE = '../../data/raw/corpus_filtered_stage0.pt'
OUTPUT_FILE = '../../data/raw/corpus_filtered_stage1.pt'
MODEL_NAME = 'path/to/your/model' # To be filled. We use Qwen3-4B-Instruct.


def worker(proc_idx: int, prompts: List[str], model_name: str, result_queue: mp.Queue):
    print(proc_idx)

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{proc_idx}'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['VLLM_LOG_LEVEL'] = 'DEBUG'
    from vllm import LLM, SamplingParams

    f = open(f'../../log/filter_stage1_{proc_idx}.log', 'w')
    sys.stdout = f
    sys.stderr = f
    logging.getLogger().setLevel(logging.CRITICAL)
    logging.getLogger("vllm").setLevel(logging.CRITICAL)
    logging.getLogger("transformers").setLevel(logging.CRITICAL)

    llm = LLM(
        model=model_name,
        max_model_len=1536,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.98,
    )

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.2,
    )

    print('Total prompts:', len(prompts))
    outs = []
    for response in llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True):
        text = response.outputs[0].text
        outs.append(text)

    result_queue.put((proc_idx, outs))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    with open(INPUT_FILE, 'rb') as f:
        all_docs = torch.load(f, weights_only=False)
    print(f'Before filtering, {len(all_docs)} data in total')

    # This prompt template is designed for Qwen series models.
    messages = [
        f"<|im_start|>system\nYou are a data quality judging assistant.<|im_end|>\n"
        f"<|im_start|>user\n### Instruction\n"
        "You will be given a document. Verify if the document is a piece of human-readable plain text.\n\n"
        "### Document\n"
        f"{document}\n\n"
        "### Criteria\n"
        "ALL PART of the document must look like a piece of descriptive text. If it seems to be a piece of code, table, figure or other non-descriptive elements, it is considered of BAD quality. You must be a strict judge and consider a text as BAD at any opportunity.\n\n" 
        "### Restriction\n"
        "1. You must generate your judgement in this STRICT format:\n"
        "<document>GOOD OR BAD</document>\n"
        "2. You must not generate any other text.<|im_end|>\n"
        "<|im_start|>assistant\n"
        for document in all_docs
    ]
    print(messages[0])

    gpu_start_index = 0
    num_procs = 8
    chunk_size = len(messages) // num_procs
    chunks = [
        messages[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_procs)
    ]
    chunks[-1].extend(messages[num_procs * chunk_size:])

    result_queue = mp.Queue()

    processes = []
    for idx in range(num_procs):
        p = mp.Process(
            target=worker,
            args=(idx, chunks[idx], MODEL_NAME, result_queue),
        )
        p.start()
        processes.append(p)

    results_by_idx = [None] * num_procs
    for _ in range(num_procs):
        idx, outs = result_queue.get()
        results_by_idx[idx] = outs

    for p in processes:
        p.join()

    final_results = []
    for lst in results_by_idx:
        final_results.extend(lst)
    print(final_results[:10])

    failed_count = 0
    bad_count = 0
    final_data = []

    for i, result in tqdm(list(enumerate(final_results)), desc="Checking and Formatting"):
        try:
            document = result[result.find('<document>') + len('<document>'): result.find('</document>')]
            if document not in ['BAD','GOOD']:
                raise ValueError("Invalid document quality")
            if document == 'BAD':
                bad_count += 1
            else:
                final_data.append(all_docs[i])
        except Exception as e:
            failed_count += 1

    print(f"{failed_count} out of {len(final_results)} failed to parse.")
    print(f"{bad_count} out of {len(final_results)} are considered bad.")
    print(f'After filtering, {len(final_data)} data left')
    torch.save(final_data, OUTPUT_FILE)