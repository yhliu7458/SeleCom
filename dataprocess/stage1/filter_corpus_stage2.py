import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import torch
import multiprocessing as mp
from typing import List
from tqdm import tqdm
import sys
sys.path.append('..')
import logging


INPUT_FILE = '../../data/raw/corpus_filtered_stage1.pt'
OUTPUT_FILE = '../../data/raw/corpus_filtered_stage2.pt'
MODEL_NAME = 'path/to/your/model' # To be filled. We use Qwen3-4B-Instruct.


def worker(proc_idx: int, prompts: List[str], model_name: str, result_queue: mp.Queue): 
    print(proc_idx)

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{proc_idx}'
    from vllm import LLM, SamplingParams

    f = open(f'../../log/filter_stage2_{proc_idx}.log', 'w')
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


if  __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    with open(INPUT_FILE, 'rb') as f:
        all_docs = torch.load(f, weights_only=False)

    messages = [
        f"<|im_start|>system\nYou are a data analysis assistant.<|im_end|>\n"
        f"<|im_start|>user\n### Instruction\n"
        "You will be given a document. Evaluate the amount of information contained by the document.\n\n"
        "### Document\n"
        f"{document}\n\n"
        "### Criteria\n"
        "You will score the amount of information of the document as an integer in 1,2,3,4,5,6,7,8,9,10.\n\n" 
        "### Format restriction\n"
        "1. You must generate your evaluation in this STRICT format:\n"
        "<response>YOUR EVALUATION</response>\n"
        "2. You must not generate any other text.<|im_end|>\n"
        "<|im_start|>assistant\n"
        for document in all_docs
    ]
    valid_doc_ids = []
    print(messages[0])
    print('Total prompts:', len(messages))

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
    final_data = []
    failed_count = 0
    bad_count = 0
    for i, result in tqdm(list(enumerate(final_results)), desc="Checking and Formatting"):
        try:
            document = result[result.find('<response>') + len('<response>'): result.find('</response>')]
            document = int(document)
            if not document in [1,2,3,4,5,6,7,8,9,10]:
                raise ValueError('Invalid score')
            if document < 6:
                bad_count += 1
            else:
                final_data.append(all_docs[i])
        except Exception as e:
            failed_count += 1

    print(f"{failed_count} out of {len(final_results)} failed to parse.")
    print(f"{bad_count} out of {len(final_results)} are considered bad.")
    print(f'After filtering, {len(final_data)} data left')
    torch.save(final_data, OUTPUT_FILE)