from tqdm import tqdm
import json
import sys
sys.path.append('../..')
from util.util import save_jsonl, load_jsonl  
import os
import random
os.environ["TOKENIZERS_PARALLELISM"] = "true"  
from transformers import AutoTokenizer


all_data = load_jsonl('../../data/qa/qda_train_new.jsonl')
random.shuffle(all_data)
all_data = all_data
print(f'Loaded {len(all_data)} data')

tokenizer = AutoTokenizer.from_pretrained('path/to/your/selector/tokenizer', padding_side='left', use_fast=True, trust_remote_code=True) # To be filled. We use Qwen3-embedding-0.6B.
generator_tokenizer = AutoTokenizer.from_pretrained('path/to/your/generator/tokenizer', padding_side='left', use_fast=True, trust_remote_code=True) # To be filled. We use Qwen3-7B-Instruct.

final_data = []
batch_size = 131072 
all_doc_len = 0
total_doc = 0

for start in tqdm(range(0, len(all_data), batch_size)):
	batch = all_data[start:start + batch_size]
	documents = [doc for item in batch for doc in item['documents']]
	document_count = [len(item['documents']) for item in batch]
	questions = [item['question'] for item in batch]
	answers = [item['answer'] for item in batch]

	document_length = tokenizer(
		documents,
		add_special_tokens=False,
		return_length=True
	)["length"]
	encoder_question_length = tokenizer(
		questions,
		add_special_tokens=False,
		return_length=True
	)["length"]
	decoder_question_length = generator_tokenizer(
		questions,
		add_special_tokens=False,
		return_length=True
	)["length"]
	decoder_answer_length = generator_tokenizer(
		answers,
		add_special_tokens=False,
		return_length=True
	)["length"]

	cummulative_count = 0
	for i, item in enumerate(batch):
		valid = True
		for doc_length in document_length[cummulative_count: cummulative_count + document_count[i]]:
			all_doc_len += doc_length
			total_doc += 1
		for doc_length in document_length[cummulative_count: cummulative_count + document_count[i]]:
			if doc_length > 2048:
				valid = False
				print('Long document!')
				break
		cummulative_count += document_count[i]
		if valid and encoder_question_length[i] > 128:
			valid = False
			print('Long question!')
			print(questions[i])
		if valid and decoder_question_length[i] > 128:
			valid = False
			print('Long question!')
			print(questions[i])
		if valid and decoder_answer_length[i] > 320:
			valid = False
			print('Long answer!')
			print(answers[i])
		if valid:
			final_data.append(item)

print(all_doc_len / total_doc)
print(f'After filtering, {len(final_data)} data left')
save_jsonl(final_data, '../../data/qa/qda_train_new_filtered.jsonl')