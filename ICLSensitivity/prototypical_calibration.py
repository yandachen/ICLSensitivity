import numpy as np
import random
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment


def prototypical_calibrate_fit(pred_probs, estimate_set_size=1000, num_gmm_runs=10):
	num_options = pred_probs.shape[1]
	log_pred_probs = np.log(pred_probs)
	assert log_pred_probs.shape == pred_probs.shape
	log_pred_probs = log_pred_probs.tolist()

	# sample estimate set
	estimate_set_example_idxs = random.sample(range(len(log_pred_probs)), min(estimate_set_size, len(log_pred_probs)))
	estimate_set_log_pred_probs = [log_pred_probs[idx] for idx in estimate_set_example_idxs]

	# fit GMM on the estimate set multiple times and choose the one with lowest "cost"
	min_cost = np.inf
	optimal_gmm = None
	gmm_cluster_label_assignment = None
	for gmm_run_idx in range(num_gmm_runs):
		# fit GMM on the estimate set
		gm = GaussianMixture(n_components=num_options, init_params='k-means++').fit(estimate_set_log_pred_probs)
		# cluster-label assignment
		belonging_scores = np.zeros((num_options, num_options))
		for cluster_idx in range(num_options):
			for label_idx in range(num_options):
				belonging_scores[cluster_idx][label_idx] = gm.means_[cluster_idx][label_idx]
		# Kuhn-Munkres algorithm to find bipartite matching
		row_ind, col_ind = linear_sum_assignment(-belonging_scores)
		assert np.all(row_ind == np.arange(num_options)) # sanity check
		cost = -belonging_scores[row_ind, col_ind].sum()
		# check if best gmm estimation so far
		if cost < min_cost:
			gmm_cluster_label_assignment = col_ind
			optimal_gmm = gm

	assert gmm_cluster_label_assignment is not None and optimal_gmm is not None
	return optimal_gmm, gmm_cluster_label_assignment


def prototypical_calibrate_predict(pc_parameters, pred_probs):
	optimal_gmm, gmm_cluster_label_assignment = pc_parameters
	num_options = pred_probs.shape[1]
	log_pred_probs = np.log(pred_probs)
	assert log_pred_probs.shape == pred_probs.shape
	log_pred_probs = log_pred_probs.tolist()
	cluster_assignments = optimal_gmm.predict(log_pred_probs)
	cluster_soft_assignments = optimal_gmm.predict_proba(log_pred_probs) # already normalized
	assert np.all(np.argmax(cluster_soft_assignments, axis=1) == cluster_assignments)
	assert cluster_soft_assignments.shape == (len(log_pred_probs), num_options)
	pred_label_probabilities = np.zeros((len(log_pred_probs), num_options))
	for example_idx in range(len(log_pred_probs)):
		for cluster_idx in range(num_options):
			pred_label_probabilities[example_idx][gmm_cluster_label_assignment[cluster_idx]] = cluster_soft_assignments[example_idx][cluster_idx]
	return pred_label_probabilities


def prototypical_calibrate_fit_predict(pred_probs, estimate_set_size):
	pc_parameters = prototypical_calibrate_fit(pred_probs, estimate_set_size)
	return prototypical_calibrate_predict(pc_parameters, pred_probs)

