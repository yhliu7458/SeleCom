import argparse
import ast
import os
import pickle
import random
import sys

import datasets
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from util.util import load_jsonl, save_jsonl


DEFAULT_DATASET_ROOT = "../data/source_datasets"
DEFAULT_OUTPUT_DIR = "../data/raw"

TRAIN_DATASETS = [
    "nq",
    "webqa",
    "wikiqa",
    "yahoo_qa",
    "freebase_qa",
    "ms_marco",
    "drop",
    "narrativeqa",
    "pubmed_qa",
    "quail",
    "squad_v2",
    "pwc",
    "triviaqa",
    "hotpotqa",
    "factkg",
]

EVAL_DATASETS = ["webqa", "nq", "hotpotqa", "popqa", "triviaqa", "factkg"]


def dataset_ref(dataset_root, dataset_name):
    local_path = os.path.join(dataset_root, dataset_name)
    if os.path.exists(local_path):
        return local_path
    return dataset_name


def load_dataset(dataset_root, dataset_name, *args):
    return datasets.load_dataset(dataset_ref(dataset_root, dataset_name), *args)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def process_question(question):
    question = str(question).strip()
    while question.endswith("?") or question.endswith("."):
        question = question[:-1].strip()
    return question + "?"


def process_answer(answer):
    if isinstance(answer, list):
        return [process_answer(item) for item in answer]

    answer = str(answer).strip()
    while answer.endswith("."):
        answer = answer[:-1].strip()
    return answer + "."


def log_dataset(name, train_data, test_data):
    print("#" * 40)
    print(f"Dataset: {name} | Train: {len(train_data)} | Test: {len(test_data)}")
    sample = train_data[0] if train_data else test_data[0]
    print(f"Sample question: {sample['question']}")
    print(f"Sample answer: {sample['answer']}")
    if sample.get("documents"):
        print(f"Sample document: {sample['documents'][0]}")


def process_nq(dataset_root):
    dataset = load_dataset(dataset_root, "nq_open")
    train_data = []
    test_data = []

    for sample in tqdm(dataset["train"], desc="nq train"):
        train_data.append(
            {
                "question": process_question(sample["question"]),
                "answer": process_answer(sample["answer"][0]),
            }
        )

    for sample in tqdm(dataset["validation"], desc="nq validation"):
        test_data.append(
            {
                "question": process_question(sample["question"]),
                "answer": process_answer(sample["answer"]),
            }
        )

    log_dataset("nq_open", train_data, test_data)
    return train_data, test_data


def process_webqa(dataset_root):
    dataset = load_dataset(dataset_root, "web_questions")
    train_data = []
    test_data = []

    for sample in tqdm(dataset["train"], desc="webqa train"):
        train_data.append(
            {
                "question": process_question(sample["question"]),
                "answer": process_answer(sample["answers"][0]),
            }
        )

    for sample in tqdm(dataset["test"], desc="webqa test"):
        test_data.append(
            {
                "question": process_question(sample["question"]),
                "answer": process_answer(sample["answers"][0]),
            }
        )

    log_dataset("web_questions", train_data, test_data)
    return train_data, test_data


def process_wikiqa(dataset_root):
    dataset = load_dataset(dataset_root, "wiki_qa")
    train_data = []
    test_data = []

    for sample in tqdm(dataset["train"], desc="wikiqa train"):
        if sample["label"] == 0:
            continue
        train_data.append(
            {
                "question": process_question(sample["question"]),
                "answer": process_answer(sample["answer"]),
            }
        )

    for sample in tqdm(dataset["test"], desc="wikiqa test"):
        if sample["label"] == 0:
            continue
        test_data.append(
            {
                "question": process_question(sample["question"]),
                "answer": process_answer(sample["answer"]),
            }
        )

    log_dataset("wiki_qa", train_data, test_data)
    return train_data, test_data


def process_yahoo_qa(dataset_root):
    dataset = load_dataset(dataset_root, "yahoo_answers_qa")
    train_data = []

    for sample in tqdm(dataset["train"], desc="yahoo_qa train"):
        train_data.append(
            {
                "question": process_question(sample["question"]),
                "answer": process_answer(sample["answer"]),
            }
        )

    log_dataset("yahoo_answers_qa", train_data, [])
    return train_data, []


def process_freebase_qa(dataset_root):
    dataset = load_dataset(dataset_root, "freebase_qa")
    train_data = []
    test_data = []

    for split_name, output in [("train", train_data), ("test", test_data)]:
        for sample in tqdm(dataset[split_name], desc=f"freebase_qa {split_name}"):
            answer = sample["Parses"]["Answers"][0]["AnswersName"][0][0]
            output.append(
                {
                    "question": process_question(sample["RawQuestion"]),
                    "answer": process_answer(answer),
                }
            )

    log_dataset("freebase_qa", train_data, test_data)
    return train_data, test_data


def process_ms_marco(dataset_root):
    dataset = load_dataset(dataset_root, "ms_marco", "v2.1")
    train_data = []

    for sample in tqdm(dataset["train"], desc="ms_marco train"):
        answer = sample["answers"][0]
        if answer == "No Answer Present.":
            continue
        train_data.append(
            {
                "question": process_question(sample["query"].lstrip(")")),
                "answer": process_answer(answer),
            }
        )

    random.shuffle(train_data)
    train_data = train_data[:100000]
    log_dataset("ms_marco", train_data, [])
    return train_data, []


def process_drop(dataset_root):
    dataset = load_dataset(dataset_root, "drop")
    train_data = []

    for sample in tqdm(dataset["train"], desc="drop train"):
        train_data.append(
            {
                "question": process_question(sample["question"]),
                "documents": [sample["passage"]],
                "answer": process_answer(sample["answers_spans"]["spans"][0]),
            }
        )

    log_dataset("drop", train_data, [])
    return train_data, []


def process_narrativeqa(dataset_root):
    dataset = load_dataset(dataset_root, "narrativeqa")
    train_data = []

    for sample in tqdm(dataset["train"], desc="narrativeqa train"):
        train_data.append(
            {
                "question": process_question(sample["question"]["text"]),
                "documents": [sample["document"]["summary"]["text"]],
                "answer": process_answer(sample["answers"][0]["text"]),
            }
        )

    log_dataset("narrativeqa", train_data, [])
    return train_data, []


def process_pubmed_qa(dataset_root):
    dataset = load_dataset(dataset_root, "pubmed_qa", "pqa_labeled")
    train_data = []

    for sample in tqdm(dataset["train"], desc="pubmed_qa train"):
        answer = sample["long_answer"] + " So the final answer is: " + sample["final_decision"]
        train_data.append(
            {
                "question": process_question(sample["question"]),
                "documents": sample["context"]["contexts"],
                "answer": process_answer(answer),
            }
        )

    log_dataset("pubmed_qa", train_data, [])
    return train_data, []


def process_quail(dataset_root):
    dataset = load_dataset(dataset_root, "quail")
    train_data = []

    for sample in tqdm(dataset["train"], desc="quail train"):
        question = process_question(sample["question"]) + "\n\n"
        for answer_id, answer in enumerate(sample["answers"]):
            question += ["A. ", "B. ", "C. ", "D. "][answer_id] + answer + "\n"
        train_data.append(
            {
                "question": question,
                "documents": [sample["context"]],
                "answer": ["A", "B", "C", "D"][sample["correct_answer_id"]],
            }
        )

    log_dataset("quail", train_data, [])
    return train_data, []


def process_squad_v2(dataset_root):
    dataset = load_dataset(dataset_root, "squad_v2")
    train_data = []

    for sample in tqdm(dataset["train"], desc="squad_v2 train"):
        if not sample["answers"]["text"]:
            continue
        train_data.append(
            {
                "question": process_question(sample["question"]),
                "documents": [sample["context"]],
                "answer": process_answer(sample["answers"]["text"][0]),
            }
        )

    log_dataset("squad_v2", train_data, [])
    return train_data, []


def process_pwc(dataset_root):
    dataset = load_dataset(dataset_root, "pwc")
    train_data = []

    for sample in tqdm(dataset["train"], desc="pwc train"):
        train_data.append(
            {
                "question": sample["prompt"],
                "documents": [sample["input"]],
                "answer": process_answer(sample["answer"]),
            }
        )

    random.shuffle(train_data)
    train_data = train_data[:100000]
    log_dataset("pwc", train_data, [])
    return train_data, []


def process_triviaqa(dataset_root):
    train_path = os.path.join(dataset_root, "triviaqa", "tqa-train.jsonl")
    test_path = os.path.join(dataset_root, "triviaqa", "test.jsonl")
    train_dataset = load_jsonl(train_path)
    test_dataset = load_jsonl(test_path)
    train_data = []
    test_data = []

    for sample in tqdm(train_dataset, desc="triviaqa train"):
        train_data.append(
            {
                "question": process_question(sample["question"]),
                "answer": process_answer(sample["answer"][0]),
            }
        )

    for sample in tqdm(test_dataset, desc="triviaqa test"):
        test_data.append(
            {
                "question": process_question(sample["question"]),
                "answer": process_answer(sample["answer"][0]),
            }
        )

    log_dataset("triviaqa", train_data, test_data)
    return train_data, test_data


def process_hotpotqa(dataset_root):
    dataset = load_dataset(dataset_root, "hotpotqa", "distractor")
    train_data = []
    test_data = []

    for split_name, output in [("train", train_data), ("validation", test_data)]:
        for sample in tqdm(dataset[split_name], desc=f"hotpotqa {split_name}"):
            contexts = sample["context"]
            titles = contexts["title"]
            contents = ["".join(sentences) for sentences in contexts["sentences"]]
            supporting_titles = set(sample["supporting_facts"]["title"])
            documents = [
                f"Title: {title}\nContent: {content}"
                for title, content in zip(titles, contents)
                if title in supporting_titles
            ]
            output.append(
                {
                    "question": process_question(sample["question"]),
                    "answer": process_answer(sample["answer"]),
                    "documents": documents,
                }
            )

    log_dataset("hotpotqa", train_data, test_data)
    return train_data, test_data


def process_popqa(dataset_root):
    dataset = load_dataset(dataset_root, "popqa")
    test_data = []

    for sample in tqdm(dataset["test"], desc="popqa test"):
        answers = ast.literal_eval(sample["possible_answers"])
        test_data.append(
            {
                "question": process_question(sample["question"]),
                "answer": [process_answer(answer) for answer in answers],
            }
        )

    log_dataset("popqa", [], test_data)
    return [], test_data


def process_factkg(dataset_root):
    train_path = os.path.join(dataset_root, "factkg", "factkg_train.pickle")
    test_path = os.path.join(dataset_root, "factkg", "factkg_test.pickle")
    train_raw = pickle.load(open(train_path, "rb"))
    test_raw = pickle.load(open(test_path, "rb"))
    train_data = []
    test_data = []

    for claim in tqdm(train_raw, desc="factkg train"):
        train_data.append(
            {
                "question": f'Verify the following claims with "True" or "False": {claim}',
                "answer": str(train_raw[claim]["Label"][0]),
            }
        )

    for claim in tqdm(test_raw, desc="factkg test"):
        test_data.append(
            {
                "question": f'Verify the following claims with "True" or "False": {claim}',
                "answer": str(test_raw[claim]["Label"][0]),
            }
        )

    random.shuffle(train_data)
    train_data = train_data[:100000]
    log_dataset("factkg", train_data, test_data)
    return train_data, test_data


PROCESSORS = {
    "nq": process_nq,
    "webqa": process_webqa,
    "wikiqa": process_wikiqa,
    "yahoo_qa": process_yahoo_qa,
    "freebase_qa": process_freebase_qa,
    "ms_marco": process_ms_marco,
    "drop": process_drop,
    "narrativeqa": process_narrativeqa,
    "pubmed_qa": process_pubmed_qa,
    "quail": process_quail,
    "squad_v2": process_squad_v2,
    "pwc": process_pwc,
    "triviaqa": process_triviaqa,
    "hotpotqa": process_hotpotqa,
    "popqa": process_popqa,
    "factkg": process_factkg,
}

EVAL_OUTPUTS = {
    "webqa": "qa_webqa_test.jsonl",
    "nq": "qa_nq_test.jsonl",
    "hotpotqa": "qda_hotpotqa.jsonl",
    "popqa": "qa_popqa_test.jsonl",
    "triviaqa": "qa_triviaqa_test.jsonl",
    "factkg": "qa_factkg_test.jsonl",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare QA-format data before attaching retrieved documents."
    )
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_output", type=str, default="qa_train.jsonl")
    parser.add_argument("--train_datasets", nargs="+", default=TRAIN_DATASETS)
    parser.add_argument("--eval_datasets", nargs="+", default=EVAL_DATASETS)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    ensure_dir(args.output_dir)

    needed_datasets = []
    if not args.skip_train:
        needed_datasets.extend(args.train_datasets)
    if not args.skip_eval:
        needed_datasets.extend(args.eval_datasets)
    needed_datasets = list(dict.fromkeys(needed_datasets))

    processed = {}
    for name in needed_datasets:
        if name not in PROCESSORS:
            raise ValueError(f"Unknown dataset: {name}")
        processed[name] = PROCESSORS[name](args.dataset_root)

    if not args.skip_train:
        full_train_data = []
        for name in args.train_datasets:
            train_data, _ = processed[name]
            full_train_data.extend(train_data)
        random.shuffle(full_train_data)
        save_jsonl(full_train_data, os.path.join(args.output_dir, args.train_output))

    if not args.skip_eval:
        for name in args.eval_datasets:
            _, test_data = processed[name]
            if name not in EVAL_OUTPUTS:
                continue
            save_jsonl(test_data, os.path.join(args.output_dir, EVAL_OUTPUTS[name]))


if __name__ == "__main__":
    main()
