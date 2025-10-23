import datasets
import os
import sys
import random
import ast
sys.path.append('../..')
from util.util import save_jsonl, load_jsonl
from tqdm import tqdm


def process_question(q):
	if q.endswith('?'):
		return process_question(q[:-1])
	elif q.endswith('.'):
		return process_question(q[:-1])
	else:
		return q.strip() + '?'

def process_answer(a):
	if not isinstance(a, list):
		a = a.strip(' ')
		if not a.endswith('.'):
			return a + '.'
		else:
			return process_answer(a[:-1])
	else:
		return [process_answer(a) for a in a]

def process_nq(path):
	dataset = datasets.load_dataset(os.path.join(path, "nq_open"))
	train_data = []
	test_data = []
	for sample in tqdm(dataset['train']):
		answer = sample['answer'][0]
		question = sample['question']
		question = process_question(question)
		answer = process_answer(answer)
		train_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)

	for sample in tqdm(dataset['validation']):
		answer = sample['answer']
		question = sample['question']
		question = process_question(question)
		answer = process_answer(answer)
		test_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: nq_open | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	return train_data, test_data

def process_commonsense_qa(path):
	dataset = datasets.load_dataset(os.path.join(path, "commonsense_qa"))
	train_data = []
	test_data = []
	for sample in tqdm(dataset['train']):
		answer = sample['answerKey']
		question = sample['question']
		question = process_question(question) +'\n\n'
		for choice,text in zip(sample['choices']['label'],sample['choices']['text']):
			question += choice + ". " + text + '\n'

		train_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: commonsense_qa | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	return train_data, test_data

def process_webqa(path):
	dataset = datasets.load_dataset(os.path.join(path, "web_questions"))
	train_data = []
	test_data = []
	for sample in tqdm(dataset['train']):
		answer = sample['answers'][0]
		question = sample['question']
		question = process_question(question)
		answer = process_answer(answer)
		train_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)
	for sample in tqdm(dataset['test']):
		answer = sample['answers'][0]
		question = sample['question']
		question = process_question(question)
		answer = process_answer(answer)
		test_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: web_questions | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	return train_data, test_data

def process_wikiqa(path):
	dataset = datasets.load_dataset(os.path.join(path, "wiki_qa"))
	train_data = []
	test_data = []
	for sample in tqdm(dataset['train']):
		if sample['label'] == 0: 
			continue
		answer = sample['answer']
		question = sample['question']
		question = process_question(question)
		answer = process_answer(answer)
		train_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)
	for sample in tqdm(dataset['test']):
		if sample['label'] == 0: 
			continue
		answer = sample['answer']
		question = sample['question']
		question = process_question(question)
		answer = process_answer(answer)
		test_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: wiki_qa | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	return train_data, test_data


def process_yahoo_qa(path):
	dataset = datasets.load_dataset(os.path.join(path, "yahoo_answers_qa"))
	train_data = []
	test_data = []
	for sample in tqdm(dataset['train']):
		answer = sample['answer']
		question = sample['question']
		question = process_question(question)
		answer = process_answer(answer)
		train_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: yahoo_answers_qa | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	return train_data, test_data

def process_freebase_qa(path):
	dataset = datasets.load_dataset(os.path.join(path, "freebase_qa"))
	train_data = []
	test_data = []
	for sample in tqdm(dataset['train']):
		answer = sample['Parses']['Answers'][0]['AnswersName'][0][0]
		question = sample['RawQuestion']
		question = process_question(question)
		answer = process_answer(answer)
		train_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)
	for sample in tqdm(dataset['test']):
		answer = sample['Parses']['Answers'][0]['AnswersName'][0][0]
		question = sample['RawQuestion']
		question = process_question(question)
		answer = process_answer(answer)
		test_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: freebase_qa | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	return train_data, test_data

def process_ms_marco(path):
	dataset = datasets.load_dataset(os.path.join(path, "ms_marco"), "v2.1")
	train_data = []
	test_data = []
	for sample in tqdm(dataset['train']):
		answer = sample['answers'][0]
		if answer == 'No Answer Present.':
			continue
		question = sample['query'].lstrip(")")
		question = process_question(question)
		answer = process_answer(answer)
		train_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: ms_marco | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	random.shuffle(train_data)
	train_data = train_data[:100000]
	return train_data, test_data

def process_drop(path):
	dataset = datasets.load_dataset(os.path.join(path, "drop"))
	train_data = []
	test_data = []
	for sample in tqdm(dataset['train']):
		answer = sample["answers_spans"]['spans'][0]
		question = sample['question']
		question = process_question(question)
		document = [sample['passage']]
		train_data.append(
			{
				"question": question,
				"documents": document,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: drop | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	print(f'Sample document: {train_data[0]["documents"][0]}')
	return train_data, test_data

def process_narrativeqa(path):
	dataset = datasets.load_dataset(os.path.join(path, "narrativeqa"))
	train_data = []
	test_data = []
	for sample in tqdm(dataset['train']):
		answer = sample["answers"][0]['text']
		question = sample['question']['text']
		question = process_question(question)
		document = [sample['document']['summary']['text']]
		train_data.append(
			{
				"question": question,
				"documents": document,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: narrativeqa | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	print(f'Sample document: {train_data[0]["documents"][0]}')
	return train_data, test_data


def process_pubmed_qa(path):
	dataset = datasets.load_dataset(os.path.join(path, "pubmed_qa", "pqa_labeled"))
	train_data = []
	test_data = []
	for sample in tqdm(dataset['train']):
		answer = sample['long_answer'] + "So the final answer is: " + sample["final_decision"]
		question = sample['question']
		question = process_question(question)
		documents = sample['context']['contexts']
		train_data.append(
			{
				"question": question,
				"documents": documents,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: pubmed_qa | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	print(f'Sample document: {train_data[0]["documents"][0]}')
	return train_data, test_data


def process_quail(path):
	dataset = datasets.load_dataset(os.path.join(path, "quail"))
	train_data = []
	test_data = []
	for sample in tqdm(dataset['train']):
		
		question = sample['question']
		question = process_question(question)
		for answer_id,answer in enumerate(sample['answers']):
			question += ["A. ","B. ","C. ","D. "][answer_id]+answer+'\n'
		answer = ["A","B","C","D"][sample["correct_answer_id"]]
		documents = [sample['context']]
		train_data.append(
			{
				"question": question,
				"documents": documents,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: quail | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	print(f'Sample document: {train_data[0]["documents"][0]}')
	return train_data, test_data

def process_squad_v2(path):
	dataset = datasets.load_dataset(os.path.join(path, "squad_v2"))
	train_data = []
	test_data = []
	for sample in tqdm(dataset['train']):
		if len(sample['answers']['text']) == 0:
			continue
		answer = sample['answers']['text'][0]
		question = sample['question']
		question = process_question(question)
		documents = [sample['context']]
		train_data.append(
			{
				"question": question,
				"documents": documents,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: squad_v2 | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	print(f'Sample document: {train_data[0]["documents"][0]}')
	return train_data, test_data


def process_pwc(path):
	dataset = datasets.load_dataset(os.path.join(path, "pwc"))
	train_data = []
	test_data = []
	for sample in tqdm(dataset['train']):
		answer = sample['answer']
		question = sample['prompt']
		documents = [sample['input']]
		train_data.append(
			{
				"question": question,
				"documents": documents,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: pwc | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	print(f'Sample document: {train_data[0]["documents"][0]}')
	random.shuffle(train_data)
	train_data = train_data[:100000]
	return train_data, test_data

def process_triviaqa(path):
	dataset = load_jsonl(os.path.join(path, "triviaqa", "tqa-train.jsonl"))
	test_dataset = load_jsonl(os.path.join(path, "triviaqa", "test.jsonl"))
	train_data = []
	test_data = []
	for sample in tqdm(dataset):
		answer = sample['answer'][0]
		question = sample['question']
		question = process_question(question)
		answer = process_answer(answer)	
		train_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)
	for sample in tqdm(test_dataset):
		answer = sample['answer'][0]
		question = sample['question']
		question = process_question(question)
		answer = process_answer(answer)	
		test_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: triviaqa | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	return train_data, test_data

def process_hotpotqa(path):
	dataset = datasets.load_dataset(os.path.join(path, "hotpotqa"), 'distractor')
	train_data = []
	test_data = []
	for sample in tqdm(dataset['train']):
		answer = sample['answer']
		question = sample['question']
		documents = sample['context']
		document_titles = documents['title']
		golden_titles = set(sample['supporting_facts']['title'])
		document_contents = [''.join(sentences) for sentences in documents['sentences']]
		documents = []
		for title, content in zip(document_titles, document_contents):
			if title in golden_titles:
				documents.append(f'Title: {title}\nContent: {content}')
		question = process_question(question)
		answer = process_answer(answer)
		train_data.append(
			{
				"question": question,
				"answer": answer,
				"documents": documents,
			}
		)
	for sample in tqdm(dataset['validation']):
		answer = sample['answer']
		question = sample['question']
		documents = sample['context']
		document_titles = documents['title']
		golden_titles = set(sample['supporting_facts']['title'])
		document_contents = [''.join(sentences) for sentences in documents['sentences']]
		documents = []
		for title, content in zip(document_titles, document_contents):
			if title in golden_titles:
				documents.append(f'Title: {title}\nContent: {content}')
		question = process_question(question)
		answer = process_answer(answer)
		test_data.append(
			{
				"question": question,
				"answer": answer,
				"documents": documents,
			}
		)
	print('#' * 40)
	print(f'Dataset: hotpotqa | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample document: {train_data[0]["documents"][0]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	return train_data, test_data

def process_popqa(path):
	dataset = datasets.load_dataset(os.path.join(path, "popqa"))
	train_data = []
	test_data = []
	for sample in tqdm(dataset['test']):
		answers = sample['possible_answers']
		answers = ast.literal_eval(answers)
		question = sample['question']
	
		question = process_question(question)
		answers = [process_answer(answer) for answer in answers]
		test_data.append(
			{
				"question": question,
				"answer": answers
			}
		)
	print('#' * 40)
	print(f'Dataset: popqa | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {test_data[0]["question"]}')
	print(f'Sample answer: {test_data[0]["answer"]}')
	return train_data, test_data
	
def process_factkg(path):
	import pickle as pkl
	
	train_data = []
	test_data = []
	dataset = pkl.load(open('../data/raw/factkg/factkg_train.pickle','rb'))
	for sample in tqdm(dataset):
		answer = str(dataset[sample]['Label'][0])
		question = f'Verify the following claims with "True" or "False": {sample}'
		train_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)
	dataset = pkl.load(open('../data/raw/factkg/factkg_test.pickle','rb'))
	for sample in tqdm(dataset):
		answer = str(dataset[sample]['Label'][0])
		question = f'Verify the following claims with "True" or "False": {sample}'
		test_data.append(
			{
				"question": question,
				"answer": answer,
			}
		)
	print('#' * 40)
	print(f'Dataset: factkg | Train: {len(train_data)} | Test: {len(test_data)}')
	print(f'Sample question: {train_data[0]["question"]}')
	print(f'Sample answer: {train_data[0]["answer"]}')
	random.shuffle(train_data)
	train_data = train_data[:100000]
	return train_data, test_data

def main():
	path = '../data/qa'
	train_nq, test_nq = process_nq(path)
	train_webqa, test_webqa = process_webqa(path)
	train_wikiqa, test_wikiqa = process_wikiqa(path)
	train_yahoo_qa, test_yahoo_qa = process_yahoo_qa(path)
	train_freebase_qa, test_freebase_qa = process_freebase_qa(path)
	train_ms_marco, test_ms_marco = process_ms_marco(path)
	train_drop, test_drop = process_drop(path)
	train_narrativeqa, test_narrativeqa = process_narrativeqa(path)
	train_pubmed_qa, test_pubmed_qa = process_pubmed_qa(path)
	train_quail, test_quail = process_quail(path)
	train_squad_v2, test_squad_v2 = process_squad_v2(path)
	train_pwc, test_pwc = process_pwc(path)
	train_triviaqa, test_triviaqa = process_triviaqa(path)
	train_hotpotqa, test_hotpotqa = process_hotpotqa(path)
	train_factkg, test_factkg = process_factkg(path)
	_, test_popqa = process_popqa(path)

	full_train_dataset = []
	full_train_dataset.extend(train_nq)
	full_train_dataset.extend(train_webqa)
	full_train_dataset.extend(train_wikiqa)
	full_train_dataset.extend(train_yahoo_qa)
	full_train_dataset.extend(train_freebase_qa)
	full_train_dataset.extend(train_ms_marco)
	full_train_dataset.extend(train_drop)
	full_train_dataset.extend(train_narrativeqa)
	full_train_dataset.extend(train_pubmed_qa)
	full_train_dataset.extend(train_quail)
	full_train_dataset.extend(train_squad_v2)
	full_train_dataset.extend(train_pwc)
	full_train_dataset.extend(train_triviaqa)
	full_train_dataset.extend(train_hotpotqa)
	full_train_dataset.extend(train_factkg)
	print(f'Full train dataset size: {len(full_train_dataset)}')
	save_jsonl(full_train_dataset, os.path.join(path, 'qa_train_new.jsonl'))
	
	save_jsonl(test_webqa, os.path.join(path, 'qa_webqa_test.jsonl'))
	save_jsonl(test_nq, os.path.join(path, 'qa_nq_test.jsonl'))
	save_jsonl(test_hotpotqa, os.path.join(path, 'qda_hotpotqa.jsonl'))
	save_jsonl(test_popqa, os.path.join(path, 'qa_popqa.jsonl'))
	save_jsonl(test_triviaqa, os.path.join(path, 'qa_triviaqa_test.jsonl'))
	save_jsonl(test_factkg, os.path.join(path, 'qa_factkg_test.jsonl'))

if __name__ == '__main__':
	main()