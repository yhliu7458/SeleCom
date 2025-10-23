import sys
import os
sys.path.append('..')
import time
import logging
import multiprocessing as mp
from typing import List

def worker(proc_idx: int,
           prompts: List[str],
           model_name: str,
           result_queue: mp.Queue): 
    
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{proc_idx}'
    from vllm import LLM, SamplingParams

    f = open(f'../../log/vllm/{proc_idx}.log', 'w')
    sys.stdout = f
    sys.stderr = f

    logging.getLogger().setLevel(logging.CRITICAL)
    logging.getLogger("vllm").setLevel(logging.CRITICAL)
    logging.getLogger("transformers").setLevel(logging.CRITICAL)

    llm = LLM(
        model=model_name,
        max_model_len=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
    )

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.7,
    )
    print('Total prompts:', len(prompts))

    outs = []
    s = time.time()
    for response in llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True):
        text = response.outputs[0].text
        outs.append(text)
    e = time.time()
    print(e - s)

    result_queue.put((proc_idx, outs))

def generate(model_name, messages, devices=[0,1,2,3,4,5,6,7]):
    num_procs = len(devices)
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
            args=(devices[idx], chunks[idx], model_name, result_queue),
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
    return final_results