from sensitivity import calculate_sensitivity
import json


def sensel(model_name_or_dir, tokenizer_name_or_dir,
		   train_data, test_data, template, device, label_bias_adjustment,
		   auxiliary_templates=None):
	sensitivity_type2sensitivity, pred_labels = calculate_sensitivity(
		model_name_or_dir, tokenizer_name_or_dir,
		train_data, test_data, template, device, label_bias_adjustment,
		auxiliary_templates=auxiliary_templates, return_pred_labels=True)
	output = {'pred_labels': pred_labels, 'confidences': {}}
	for sensitivity in sensitivity_type2sensitivity:
		output['confidences'][sensitivity] = sensitivity_type2sensitivity[sensitivity]
	return output


if __name__ == '__main__':
	templates = json.load(open('data/templates.json'))
	model_name = 'EleutherAI/gpt-neo-2.7B'
	for lba in ['none', 'cc', 'pc']:
		output = calculate_sensitivity(model_name_or_dir=model_name,
									   tokenizer_name_or_dir=model_name,
									   train_data=json.load(open('data/train.json')),
									   test_data=json.load(open('data/test.json')),
									   template=templates[0], device='cuda:0',
									   label_bias_adjustment=lba,
									   auxiliary_templates=templates[1:])
