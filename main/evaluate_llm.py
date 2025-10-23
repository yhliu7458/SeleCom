import sys
sys.path.append('..')
from util.util import load_jsonl, evaluate_exact_match, evaluate_f1, evaluate_llm_as_a_judge, evaluate_match

all_response = load_jsonl('../log/qwen/qa-hotpotqa-top5.jsonl')
try:
    all_questions = [item['question'] for item in all_response]
except:
    all_questions = [item['question'] for item in load_jsonl('../data/qa/qda_hotpotqa.jsonl')]

all_outputs = [item['output'] for item in all_response]
all_groundtruths = [item['groundtruth'] for item in all_response]

evaluate_exact_match(all_outputs, all_groundtruths)
evaluate_f1(all_outputs, all_groundtruths)
evaluate_match(all_outputs, all_groundtruths)
evaluate_llm_as_a_judge(all_questions, all_outputs, all_groundtruths)