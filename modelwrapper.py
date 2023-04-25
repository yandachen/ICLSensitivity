from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import numpy as np
from datasets import Dataset
from torch.nn import CrossEntropyLoss
from scipy.special import softmax
from tqdm import tqdm


class ModelWrapper:
	def __init__(self, model_name_or_dir, tokenizer_name_or_dir, device):
		self.model = AutoModelForCausalLM.from_pretrained(model_name_or_dir)
		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_dir)
		self.tokenizer.pad_token = self.tokenizer.eos_token
		self.tokenizer.padding_side = 'left'
		self.device = device
		self.model.to(device)

	@staticmethod
	def _collate_fn_pad(batch_examples, pad_token_id):
		"""
		Batching and padding. Nothing else.
		batch_examples is a HF Dataset with feature "input_ids"
		For CLM, assume that the batch_examples in the parameter are the prompt.
		"""
		num_examples = len(batch_examples)
		# pad to max_seq_length
		max_seq_length = max([len(example['input_ids']) for example in batch_examples])
		input_ids = torch.full((num_examples, max_seq_length), pad_token_id)
		attention_mask = torch.full((num_examples, max_seq_length), 0)
		position_ids = torch.full((num_examples, max_seq_length), 0)
		labels = torch.full((num_examples, max_seq_length), -100)
		for ex_idx, example in enumerate(batch_examples):
			word_ids = example['input_ids']
			ex_len = len(word_ids)
			input_ids[ex_idx][-ex_len:] = torch.LongTensor(word_ids)
			attention_mask[ex_idx][-ex_len:] = 1
			position_ids[ex_idx][-ex_len:] = torch.arange(ex_len)
			labels[ex_idx][-ex_len:] = torch.LongTensor(example['labels'])
		return input_ids, attention_mask, position_ids, labels

	def score_options(self, data, max_seq_length, bsz):
		"""
		eval_data is a HF Dataset with two features 'prompt_input_ids', 'options_input_ids'
		"""
		assert data.column_names == ['prompt_input_ids', 'options_input_ids']

		input_ids, labels = [], []
		for example in data:
			prompt_input_ids, options_input_ids = example['prompt_input_ids'], example['options_input_ids']
			for option_input_ids in options_input_ids:
				input_ids.append(prompt_input_ids + option_input_ids)
				labels.append([-100] * len(prompt_input_ids) + option_input_ids)
		input_ids = [ex[-max_seq_length:] for ex in input_ids]
		labels = [ex[-max_seq_length:] for ex in labels]
		eval_data = Dataset.from_dict({'input_ids': input_ids, 'labels': labels})
		eval_dataloader = DataLoader(eval_data, bsz, shuffle=False,
									 collate_fn=lambda batch_examples:
									 ModelWrapper._collate_fn_pad(batch_examples, self.tokenizer.pad_token_id))
		self.model.eval()

		eval_data_losses = []
		loss_fct = CrossEntropyLoss(reduction='sum')  # no length normalization is applied.
		for batch_data in tqdm(eval_dataloader):
			input_ids, attention_mask, position_ids, labels = batch_data
			input_ids, attention_mask, position_ids, labels = \
				input_ids.to(self.device), attention_mask.to(self.device), position_ids.to(self.device), labels.to(
					self.device)
			with torch.no_grad():
				output = self.model.forward(input_ids=input_ids, attention_mask=attention_mask,
											position_ids=position_ids)
				batch_losses = []
				for ex_idx in range(len(input_ids)):
					ex_loss = loss_fct(output.logits[ex_idx][: -1], labels[ex_idx][1:])
					batch_losses.append(ex_loss)
				batch_losses = torch.stack(batch_losses)
			eval_data_losses += batch_losses.cpu().tolist()

		assert len(eval_data_losses) == len(eval_data), (len(eval_data_losses), len(eval_data))

		pred_probs = []
		cur_ptr = 0
		for ex_idx in range(len(data)):
			num_options = len(data[ex_idx]['options_input_ids'])
			ex_pred_loss = eval_data_losses[cur_ptr: cur_ptr + num_options]
			pred_probs.append(softmax(-np.array(ex_pred_loss)))
			cur_ptr += num_options
		assert cur_ptr == len(eval_data_losses)
		assert len(pred_probs) == len(data)
		return np.array(pred_probs)

