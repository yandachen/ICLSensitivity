from modelwrapper import ModelWrapper
from datasets import Dataset
import json
from prototypical_calibration import prototypical_calibrate_fit_predict
import numpy as np


def _icl(model_name_or_dir, tokenizer_name_or_dir, train_data, test_data, template, device):
	# prepare prompts
	instruction, verbalizers = template['instruction'], template['verbalizers']
	assert instruction.endswith('<verbalizer>')  # autoregressive prediction
	num_labels = len(verbalizers)
	fewshot_prefix = ''
	for train_ex in train_data:
		example = instruction.replace('<text>', train_ex['text']).replace('<verbalizer>',
																		  verbalizers[train_ex['label']])
		fewshot_prefix += example + '\n\n'
	prompts, options = [], []
	for test_ex in test_data:
		if instruction.endswith(' <verbalizer>'):
			prompt = fewshot_prefix + instruction.replace('<text>', test_ex['text']).replace(' <verbalizer>', '')
		else:
			prompt = fewshot_prefix + instruction.replace('<text>', test_ex['text']).replace('<verbalizer>', '')
		prompts.append(prompt)
		opts = []
		for test_label in range(num_labels):
			option = ' ' + verbalizers[test_label]
			opts.append(option)
		options.append(opts)
	assert len(prompts) == len(options) == len(test_data)

	# prepare model
	mw = ModelWrapper(model_name_or_dir=model_name_or_dir,
					  tokenizer_name_or_dir=tokenizer_name_or_dir,
					  device=device)
	# tokenize
	prompt_input_ids = mw.tokenizer(prompts)['input_ids']
	option_input_ids = [mw.tokenizer(opts)['input_ids'] for opts in options]
	data = Dataset.from_dict({'prompt_input_ids': prompt_input_ids, 'options_input_ids': option_input_ids})
	pred_probs = mw.score_options(data, max_seq_length=1024, bsz=8)
	return pred_probs


def icl_with_label_bias_adjustment(model_name_or_dir, tokenizer_name_or_dir,
								   train_data, test_data, template, device,
								   label_bias_adjustment):
	assert label_bias_adjustment in ['none', 'cc', 'pc']
	if label_bias_adjustment == 'none':
		pred_probs = _icl(model_name_or_dir=model_name_or_dir, tokenizer_name_or_dir=tokenizer_name_or_dir,
						  train_data=train_data, test_data=test_data, template=template, device=device)
	elif label_bias_adjustment == 'cc':
		neutral_test_data = [{'text': ''}, {'text': '[MASK]'}, {'text': 'N/A'}]
		neutral_pred_probs = _icl(model_name_or_dir=model_name_or_dir, tokenizer_name_or_dir=tokenizer_name_or_dir,
								  train_data=train_data, test_data=neutral_test_data, template=template, device=device)
		neutral_pred_prob = np.mean(neutral_pred_probs, axis=0)
		pred_probs = _icl(model_name_or_dir=model_name_or_dir, tokenizer_name_or_dir=tokenizer_name_or_dir,
						  train_data=train_data, test_data=test_data, template=template, device=device)
		cc_pred_probs = pred_probs / neutral_pred_prob
		# re-normalize to sum up to one
		pred_probs = cc_pred_probs / np.sum(cc_pred_probs, axis=1, keepdims=True)
	elif label_bias_adjustment == 'pc':
		pred_probs = _icl(model_name_or_dir=model_name_or_dir, tokenizer_name_or_dir=tokenizer_name_or_dir,
						  train_data=train_data, test_data=test_data, template=template, device=device)
		pred_probs = prototypical_calibrate_fit_predict(pred_probs, estimate_set_size=len(pred_probs))
	else:
		raise NotImplementedError
	pred_labels = np.argmax(pred_probs, axis=1)
	return pred_labels, pred_probs


if __name__ == '__main__':
	for lba in ['none', 'cc', 'pc']:
		pred_labels, pred_probs = icl_with_label_bias_adjustment(model_name_or_dir='EleutherAI/gpt-neo-2.7B',
													tokenizer_name_or_dir='EleutherAI/gpt-neo-2.7B',
													train_data=json.load(open('data/train.json')),
													test_data=json.load(open('data/test.json')),
													template={
														'instruction': 'Is this product review positive?\nReview: <text>\nAnswer: <verbalizer>',
														'verbalizers': ['no', 'yes']},
													device='cuda:0', label_bias_adjustment=lba)
		gt_labels = [ex['label'] for ex in json.load(open('data/test.json'))]
		from sklearn.metrics import accuracy_score
		print(accuracy_score(gt_labels, pred_labels))
		print(pred_probs[:5])
