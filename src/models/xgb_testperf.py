import os 
import xgboost
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score

test_keys = os.listdir('../../data/casp_data/test/test_final_cnn_hom-hmms-FULL-sxl')

xgb_model_latest = xgboost.XGBClassifier() 
xgb_model_latest.load_model("saved_models/model_xgb_F1000_mod_hom-hmms-FULL.json")

y_true_all = []
y_pred_all = []

# metrics 
prec_all = []
rec_all = []
acc_all = []
aucroc_all = []

def get_datapoint(set_name, key):
	filename = '../../data/casp_data/test/test_final_cnn_hom-hmms-FULL-sxl/' + key 

	X_dict = {}

	arr = np.load(filename, allow_pickle=True)['arr_0']
	X_dict['cmap'] = arr.item()['cmap']
	X_dict['ft_vec_1d'] = arr.item()['ft_vec_1d']
	# print(X_dict['ft_vec_1d'])
	y = arr.item()['y']

	return X_dict, y

def weighted_cross_entropy(y_true, y_pred):
    weighting = 5
    loss_pos = y_true * np.log(y_pred)
    loss_neg = weighting * (1 - y_true) * np.log(1 - y_pred)
    loss = -1 * (loss_pos + loss_neg)
    # for j in range(len(y_true)):
    # 	print(y_true[i], y_pred[i], loss_neg[i], loss_pos[i], loss[i])
    return np.sum(loss)

pdom = np.load('../../data/casp_data/test/features_final/pdom_proc-hmms-FULL_Test.npz', allow_pickle=True)['arr_0']
print(pdom.shape)

for i in range(len(test_keys)):
	try:
		print(i, len(test_keys))
		X, y_true_sample = get_datapoint('test', test_keys[i])
		X_ft_1d_vec = X['ft_vec_1d']
		X_cmap = X['cmap']
		X_per_res = []

		for j in range(len(y_true_sample)):
			vec_x = np.asarray([])
			vec_x = np.append(vec_x, X_ft_1d_vec[j], axis=0)
			vec_x = np.append(vec_x, [np.sum(X_cmap[j]) / len(y_true_sample)], axis=0)
			X_per_res.append(vec_x)

		X_per_res = np.asarray(X_per_res)
		y_pred_sample = xgb_model_latest.predict(X_per_res)
		y_pred_sample_prob = xgb_model_latest.predict_proba(X_per_res)
		# print(X_cmap[0].shape, y_true_sample.shape, X_per_res.shape, y_pred_sample.shape, len(y_true_sample))

		y_true_all.append(y_true_sample)
		y_pred_all.append(y_pred_sample)
		for j in range(len(y_true_sample)):
			print(y_true_sample[j], y_pred_sample[j], y_pred_sample_prob[j], pdom[i][j])
		prec = precision_score(y_true_sample, np.round(y_pred_sample))
		rec = recall_score(y_true_sample, np.round(y_pred_sample))
		acc = accuracy_score(y_true_sample, np.round(y_pred_sample))
		aucroc = roc_auc_score(y_true_sample, y_pred_sample)
		print(prec, rec, acc, aucroc)
		# for j in range(len(y_true_sample)):
		# 	print(y_pred_sample[j], y_true_sample[j], np.round(y_pred_sample[j]))
		# print(weighted_cross_entropy(y_true_sample, y_pred_sample))
		# print(weighted_cross_entropy(y_true_sample, np.ones(len(y_true_sample)) - 0.01))
		# print(weighted_cross_entropy(y_true_sample, np.zeros(len(y_true_sample)) + 0.01))

		prec_all.append(prec)
		rec_all.append(rec)
		acc_all.append(acc)
		aucroc_all.append(aucroc)
		# break
	except:
		print("PASS")
		pass

print("Precision: ", np.mean(prec_all))
print("Recall: ", np.mean(rec_all))
print("Accuracy: ", np.mean(acc_all)) 
print("AUC-ROC Score: ", np.mean(aucroc_all))

'''Full Data + (XGB:n_estimators=1000, learning_rate=0.01, max_depth=7, scale_pos_weight=0.1, verbosity=2)
## Validating ##
Precision:  0.9726361194608448
Recall:  0.8357845534796046
Accuracy:  0.8238266252333724
AUC-ROC:  0.7386536479334502

## Testing ##
Precision:  0.8111922049937035
Recall:  0.7780783682852411
Accuracy:  0.7674298795944628
AUC-ROC Score:  0.7315296302437427
'''

'''Full Data with CRH
## Validating ##
Precision:  0.9835605669407856
Recall:  0.6747760061547085
Accuracy:  0.6842113098007108
AUC-ROC:  0.7514163378549684

## Testing ##
Precision:  0.5260107864751462
Recall:  0.3780256480522253
Accuracy:  0.4817198265296856
AUC-ROC Score:  0.6399739567975563
'''

'''Full Data with CRH-HMMs
Precision:  0.5600513489288786
Recall:  0.4208270955034474
Accuracy:  0.5024066532675238
AUC-ROC Score:  0.6357333333972967
'''