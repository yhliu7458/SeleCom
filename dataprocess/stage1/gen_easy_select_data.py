import random
import torch
import multiprocessing as mp
from tqdm import tqdm
import sys
sys.path.append('../..')
from util.util import save_jsonl
from vllm_client import generate


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    input_file = '../../data/raw/corpus_filtered_stage2.pt'
    output_file = '../../data/select/select-data-easy.jsonl'
    sample_size = 10000000
    model_name = 'path/to/your/model' # To be filled. We use Qwen3-4B-Instruct.

    with open(input_file, 'rb') as f:
        all_docs = torch.load(f, weights_only=False)
    sampled = random.sample(all_docs, sample_size)

    messages = [
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n### Document\n{doc}\n\n### Instruction\n"
        "Ask a question about one single fact mentioned in the above document "
        "and provide its answer. The answer should be short, concise and factual (1-10 words typically) and correctly answers the question.\n\n"
        "### Restriction\n"
        "1. You must use English.\n"
        "2. You must generate the question and its answer in this STRICT format:\n"
        "<question>YOUR QUESTION</question><answer>YOUR ANSWER</answer>\n"
        "3. You must not generate any other text.<|im_end|>\n"
        "<|im_start|>assistant\n"
        for doc in sampled
    ]
    print(messages[0])

    final_results = generate(model_name, messages)

    outputs = []
    failed_count = 0
    for i, result in tqdm(list(enumerate(final_results)), desc="Checking and Formatting"):
        try:
            question = result[result.find('<question>') + len('<question>'): result.find('</question>')]
            answer = result[result.find('<answer>') + len('<answer>'): result.find('</answer>')]
            if not question or not answer:
                raise ValueError("Empty question or answer")
            outputs.append({
                'question': question,
                'document': sampled[i],
                'answer':   answer
            })
        except Exception as e:
            print(f'Error: {e} | Raw output: {result}' )
            failed_count += 1

    print(f"{failed_count} out of {len(final_results)} failed to parse.")
    save_jsonl(outputs, output_file)
    print(f"Done: generated {len(final_results)} items, saved to {output_file}")