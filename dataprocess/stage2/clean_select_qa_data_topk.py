from tqdm import tqdm
import json
import sys
sys.path.append('..')
from util.util import save_jsonl, load_jsonl  
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"  
from transformers import AutoTokenizer

all_data = load_jsonl('../../data/qa/qda_train_filtered.jsonl')
tokenizer = AutoTokenizer.from_pretrained('path/to/your/selector/tokenizer', padding_side='left', use_fast=True, trust_remote_code=True) # To be filled. We use Qwen3-embedding-0.6B.
generator_tokenizer = AutoTokenizer.from_pretrained('path/to/your/generator/tokenizer', padding_side='left', use_fast=True, trust_remote_code=True) # To be filled. We use Qwen3-7B-Instruct.
print(f'Loaded {len(all_data)} data')


final_data = []
batch_size = 16384  
for start in tqdm(range(0, len(all_data), batch_size)):
	batch = all_data[start:start + batch_size]
	documents = [doc for item in batch for doc in item['documents'][:5]]
	document_count = [len(item['documents'][:10]) for item in batch]
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
		current_count = 0
		current_doc_length = 0
		for doc_length in document_length[cummulative_count: cummulative_count + document_count[i]]:
			current_doc_length += doc_length
			if current_doc_length > 2560:
				if current_count == 0:
					valid = False
				print('Long document!')
				break
			current_count += 1
		
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
			final_data.append({
				'question': item['question'],
				'documents': item['documents'][:current_count],
				'answer': item['answer']
			})
		
print(f'After filtering, {len(final_data)} data left')
save_jsonl(final_data, '../../data/qa/qda_train_filtered_top10.jsonl')