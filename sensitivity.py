from icl import icl_with_label_bias_adjustment
import numpy as np
from automatic_inst_perturbation import perturb_instructions
from itertools import permutations
import random
import math
import json


# icl: InstH, InstA, ExOrd
# icl parameters: model_name_or_dir, tokenizer_name_or_dir, train_data, test_data, template, device, label_bias_adjustment
# auxiliary templates

def pred_difference(pred_labels1, pred_labels2):
	return np.mean(np.array(pred_labels1) != np.array(pred_labels2))


def calculate_sensitivity(model_name_or_dir, tokenizer_name_or_dir,
						  train_data, test_data, template, device, label_bias_adjustment,
						  auxiliary_templates=None, return_pred_labels=False):
	sensitivity_type2sensitivity = {}
	pred_labels, _ = icl_with_label_bias_adjustment(model_name_or_dir, tokenizer_name_or_dir,
													train_data, test_data, template, device,
													label_bias_adjustment)
	# Inst-H
	print('calculating Inst-H')
	if auxiliary_templates is not None:
		sensitivitys = []
		for aux_template in auxiliary_templates:
			aux_pred_labels, _ = icl_with_label_bias_adjustment(model_name_or_dir, tokenizer_name_or_dir,
																train_data, test_data, aux_template, device,
																label_bias_adjustment)
			sensitivitys.append(pred_difference(pred_labels, aux_pred_labels))
		sensitivity_type2sensitivity['Inst-H'] = sensitivitys

	# Inst-A
	print('calculating Inst-A')
	perturbed_instructions = perturb_instructions(template['instruction'])
	sensitivitys = []
	for perturbed_inst in perturbed_instructions:
		perturbed_template = {'instruction': perturbed_inst, 'verbalizers': template['verbalizers']}
		perturbed_pred_labels, _ = icl_with_label_bias_adjustment(model_name_or_dir, tokenizer_name_or_dir,
																  train_data, test_data, perturbed_template, device,
																  label_bias_adjustment)
		sensitivitys.append(pred_difference(pred_labels, perturbed_pred_labels))
	sensitivity_type2sensitivity['Inst-A'] = sensitivitys

	# ExOrd
	print('calculating ExOrd')
	num_demonstrations = len(train_data)
	NUM_ORDER_PERMUTATIONS = 5
	if math.factorial(num_demonstrations) <= NUM_ORDER_PERMUTATIONS * 30:
		all_permutations = list(permutations(range(num_demonstrations)))
		all_permutations.remove(tuple(list(range(num_demonstrations)))) # original ordering
		ex_orders = random.sample(all_permutations, min(NUM_ORDER_PERMUTATIONS, len(all_permutations)))
	else:
		ex_orders = set()
		while len(ex_orders) < NUM_ORDER_PERMUTATIONS:
			order = list(range(num_demonstrations))
			random.shuffle(order)
			if order not in ex_orders and order != list(range(num_demonstrations)):
				ex_orders.add(order)
	sensitivitys = []
	for ex_order in ex_orders:
		perturbed_train_data = [train_data[idx] for idx in ex_order]
		perturbed_pred_labels, _ = icl_with_label_bias_adjustment(model_name_or_dir, tokenizer_name_or_dir,
																  perturbed_train_data, test_data, template, device,
																  label_bias_adjustment)
		sensitivitys.append(pred_difference(pred_labels, perturbed_pred_labels))
	sensitivity_type2sensitivity['ExOrd'] = sensitivitys

	if not return_pred_labels:
		return sensitivity_type2sensitivity
	else:
		return sensitivity_type2sensitivity, pred_labels


if __name__ == '__main__':
	templates = json.load(open('data/templates.json'))
	model_name = 'EleutherAI/gpt-neo-2.7B'
	for lba in ['none', 'cc', 'pc']:
		sensitivity_type2sensitivity = calculate_sensitivity(model_name_or_dir=model_name,
															 tokenizer_name_or_dir=model_name,
															 train_data=json.load(open('data/train.json')),
															 test_data=json.load(open('data/test.json')),
															 template=templates[0], device='cuda:0',
															 label_bias_adjustment=lba,
															 auxiliary_templates=templates[1:])