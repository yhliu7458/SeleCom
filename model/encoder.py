import sys
sys.path.append('..')

import torch.nn.functional as F
import torch
import torch.nn as nn
from util.constant import *
from transformers import AutoModel, AutoTokenizer

	
class Encoder(nn.Module):
	def __init__(self, args):
		super(Encoder, self).__init__()
		self.max_length = args.encoder_max_length
		self.num_emb_tokens = args.num_emb_tokens
		
		self.tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, padding_side='left', use_fast=True, trust_remote_code=True)
		self.tokenizer.add_tokens([ENCODE_TOKEN], special_tokens=True)
		self.encoder = AutoModel.from_pretrained(args.encoder_name, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
		
		self.encode_token_id = self.tokenizer.convert_tokens_to_ids(ENCODE_TOKEN)
		self.encoder.resize_token_embeddings(len(self.tokenizer))
		self.embedding_layer = self.encoder.get_input_embeddings()
		# Embedding layer for ENCODE token
		self.encode_token_embedding_layer = nn.Embedding(1, self.encoder.config.hidden_size)
		self.encode_token_embedding_layer.weight.data = self.embedding_layer.weight[self.encode_token_id].unsqueeze(0)
	
	
	def forward(self, encoder_input_ids, encoder_attention_mask):
		encoder_input_ids = encoder_input_ids.to(self.encoder.device)
		encoder_attention_mask = encoder_attention_mask.to(self.encoder.device)
		input_embeds = self.embedding_layer(encoder_input_ids)
		# Replace ENCODE token embeddings with special embedding layer
		input_embeds[encoder_input_ids == self.encode_token_id] = self.encode_token_embedding_layer.weight[0]

		model_output = self.encoder(
			inputs_embeds=input_embeds, 
			attention_mask=encoder_attention_mask
		)

		output = model_output.last_hidden_state[encoder_input_ids == self.encode_token_id]
		output = output.reshape([-1, self.num_emb_tokens * output.shape[-1]])
		
		return output
