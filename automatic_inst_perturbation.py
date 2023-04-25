from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import random
import json


class Paraphraser():
	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
		self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").cuda()

	def paraphrase(self, text, top_p=0.9, num=10):
		assert '\n' not in text  # T5 does not handle \n characters
		text = "paraphrase: " + text + " </s>"
		encoding = self.tokenizer.encode_plus(text, padding=True, return_tensors="pt")
		input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

		outputs = self.model.generate(input_ids=input_ids.cuda(),
									  attention_mask=attention_masks.cuda(),
									  max_length=256, do_sample=True, top_k=120, top_p=top_p,
									  early_stopping=True, num_return_sequences=num)
		paraphrased_texts = []
		for output in outputs:
			line = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
			paraphrased_texts.append(line)
		return paraphrased_texts


def paraphrase(paraphraser, instruction, num, top_p):
	# check that instruction is of the correct format
	assert instruction.count('<text>') == 1 and instruction.count('<verbalizer>') == 1 \
		   and instruction.endswith('<verbalizer>')
	# use special tokens as placeholders before paraphrasing
	instruction = instruction.replace('<text>', '[0]').replace('<verbalizer>', '[1]')
	# split the concatenated_text by \n breaks because T5-paraphraser only take one-line inputs
	newline_idxs = [i for i in range(len(instruction)) if instruction[i] == '\n']
	if len(newline_idxs) == 0:
		parts = [instruction]
	else:
		parts = [instruction[0: newline_idxs[0]], instruction[newline_idxs[0]]]
		for idx in range(len(newline_idxs) - 1):
			parts.append(instruction[newline_idxs[idx] + 1: newline_idxs[idx + 1]])
			parts.append(instruction[newline_idxs[idx + 1]])
		if newline_idxs[-1] != len(instruction) - 1:
			parts.append(instruction[newline_idxs[-1] + 1:])
		parts = [x for x in parts if len(x) != 0]  # remove empty strings
	# each part is either a single '\n' or does not contain '\n'
	# start paraphrasing!
	paraphrased_parts = []
	num_paraphrase_trials = num * 3
	for part in parts:
		assert '\n' not in part or part == '\n'
		if part == '\n':
			paraphrased_parts.append([part] * num_paraphrase_trials)
		else:
			model_outputs = paraphraser.paraphrase(part, top_p=top_p, num=num_paraphrase_trials)
			assert len(model_outputs) == num_paraphrase_trials
			paraphrased_parts.append(model_outputs)
	# merge paraphrased parts and count number of legal paraphrases
	legal_paraphrases = set()
	for idx in range(num_paraphrase_trials):
		paraphrased_text = ''.join([paraphrased_part[idx] for paraphrased_part in paraphrased_parts]).strip()
		if paraphrased_text == instruction or paraphrased_text.count('[0]') != 1 \
				or paraphrased_text.count('[1]') != 1 or not paraphrased_text.endswith('[1]'):
			continue
		paraphrased_text = paraphrased_text.strip().replace('[0]', '<text>').replace('[1]', '<verbalizer>')
		legal_paraphrases.add(paraphrased_text.strip())
	return legal_paraphrases


def word_dropout(instruction, word_dropout_prob, num):
	tokens = re.split(r'(\s+)', instruction) # consecutive whitespaces may not be fully separated
	assert '<text>' in tokens and '<verbalizer>' in tokens
	word_dropout_instructions = set()
	for _ in range(num * 3): # number of sampling trials
		kept_tokens = []
		for token in tokens:
			if token in ['<text>', '<verbalizer>']:
				kept_tokens.append(token)
			elif token.isspace() is False:
				p = random.random()
				if p > word_dropout_prob:
					kept_tokens.append(token)
			else:
				assert token.isspace()
				whitespaces = list(token)
				num_whitespaces = len(whitespaces)
				selected_whitespaces = [whitespaces[idx] for idx in range(num_whitespaces) if random.random() > word_dropout_prob]
				if selected_whitespaces == []:
					selected_whitespaces = [' ']
				kept_tokens.append(''.join(selected_whitespaces))
		word_dropout_inst = re.sub(' +', ' ', ''.join(kept_tokens).strip()) # remove consecutive spaces
		if word_dropout_inst != instruction:
			word_dropout_instructions.add(word_dropout_inst)
		if len(word_dropout_instructions) == num:
			break
	return word_dropout_instructions


def perturb_instructions(instruction):
	paraphraser = Paraphraser()
	paraphrase_insts = paraphrase(paraphraser, instruction, top_p=0.9, num=3)
	word_dropout_instructions = word_dropout(instruction, word_dropout_prob=0.2, num=3)
	return list(paraphrase_insts.union(word_dropout_instructions))


if __name__ == '__main__':
	instruction = 'Is this product review positive?\nReview: <text>\nAnswer: <verbalizer>'
	print(perturb_instructions(instruction))
