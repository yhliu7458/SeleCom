import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from util.constant import *

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.generator_name = args.generator_name
        self.max_length = args.generator_max_length

        self.generate_model = AutoModelForCausalLM.from_pretrained(self.generator_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        self.tokenizer = AutoTokenizer.from_pretrained(args.generator_name, padding_side='left', use_fast=True, trust_remote_code=True)
        # RANK_TOKEN is an example of special tokens for future extensions
        self.tokenizer.add_tokens([SOFT_PROMPT_START, SOFT_PROMPT_TOKEN, SOFT_PROMPT_END, RANK_TOKEN], special_tokens=True)

        self.soft_prompt_token_id = self.tokenizer.convert_tokens_to_ids(SOFT_PROMPT_TOKEN)
        self.soft_prompt_start_id = self.tokenizer.convert_tokens_to_ids(SOFT_PROMPT_START)
        self.soft_prompt_end_id = self.tokenizer.convert_tokens_to_ids(SOFT_PROMPT_END)
        self.rerank_token_id = self.tokenizer.convert_tokens_to_ids(RANK_TOKEN)

        self.generate_model.resize_token_embeddings(len(self.tokenizer))
        self.embedding_layer = self.generate_model.get_input_embeddings()

        # Embedding layers for special token
        self.soft_prompt_start_embedding_layer = nn.Embedding(1, self.embedding_layer.weight.shape[1])
        self.soft_prompt_start_embedding_layer.weight.data = self.embedding_layer.weight[self.soft_prompt_start_id].unsqueeze(0)

        self.soft_prompt_end_embedding_layer = nn.Embedding(1, self.embedding_layer.weight.shape[1])
        self.soft_prompt_end_embedding_layer.weight.data = self.embedding_layer.weight[self.soft_prompt_end_id].unsqueeze(0)

        self.rerank_token_embedding_layer = nn.Embedding(1, self.embedding_layer.weight.shape[1])
        self.rerank_token_embedding_layer.weight.data = self.embedding_layer.weight[self.rerank_token_id].unsqueeze(0)

    # Prepare input embeddings by replacing special tokens with their corresponding embeddings
    def prepare_input(self, embeddings, generator_input_ids, generator_attention_mask):
        generator_input_ids = generator_input_ids.to(self.generate_model.device)
        generator_attention_mask = generator_attention_mask.to(self.generate_model.device)

        input_embeds = self.embedding_layer(generator_input_ids)
        input_embeds[generator_input_ids == self.soft_prompt_start_id] = self.soft_prompt_start_embedding_layer.weight[0]
        input_embeds[generator_input_ids == self.soft_prompt_end_id] = self.soft_prompt_end_embedding_layer.weight[0]
        input_embeds[generator_input_ids == self.rerank_token_id] = self.rerank_token_embedding_layer.weight[0]
        input_embeds[generator_input_ids == self.soft_prompt_token_id] = embeddings
        
        return input_embeds, generator_attention_mask

    def forward(self, embeddings, generator_input_ids, generator_attention_mask, generator_labels, return_logits=False, return_loss=True):
        input_embeds, input_mask = self.prepare_input(embeddings, generator_input_ids, generator_attention_mask)
        generator_labels = generator_labels.to(self.generate_model.device)

        if return_loss:
            output = self.generate_model.forward(
                input_ids=None,
                attention_mask=input_mask,
                inputs_embeds=input_embeds,
                labels=generator_labels
            )
            if return_logits:
                answer_mask = generator_labels != IGNORE_TOKEN_ID
                return output.loss, output.logits[answer_mask]
            else:
                return output.loss
        else:
            assert return_logits is True
            output = self.generate_model.forward(
                input_ids=None,
                attention_mask=generator_attention_mask,
                inputs_embeds=input_embeds
            )
            answer_mask = generator_labels != IGNORE_TOKEN_ID
            return output.logits[answer_mask]

    def generate(self, embeddings, generator_input_ids, generator_attention_mask):
        input_embeds, input_mask = self.prepare_input(embeddings, generator_input_ids, generator_attention_mask)

        output = self.generate_model.generate(
            input_ids=None,
			attention_mask=input_mask,
			inputs_embeds=input_embeds,
            max_new_tokens=1024,
            temperature=0.2,
            use_cache=True
        )
    
        generated_text = []
        for o in output:
            generated_text.append(self.tokenizer.decode(o, skip_special_tokens=True))
            
        return generated_text





    