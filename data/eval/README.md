# Evaluation Data

The evaluation JSONL files are released through Hugging Face instead of this Git repository.

Hugging Face dataset: [Ryan7458/Eval_QDA](https://huggingface.co/datasets/Ryan7458/Eval_QDA)

Expected files:

- `qda_factkg_multidoc.jsonl`
- `qda_hotpotqa.jsonl`
- `qda_nq_multidoc.jsonl`
- `qda_popqa_multidoc.jsonl`
- `qda_triviaqa_multidoc.jsonl`
- `qda_webqa_multidoc.jsonl`

For single-document evaluation settings, use the first retrieved document in `documents`, i.e., `documents[0]`.
