import numpy as np 
import pandas as pd 

'''
Test Max: 2180
Train-Val Max: 3042
'''

ds = pd.read_csv('train-val/ds_final_train-val.csv')
dom_annots = list(ds["chain-domain-annots "])
chain_desc = list(ds["chain-desc"])

max_res = 3042

# t5
t5 = np.load('train-val/DomDet_Res_T5.npz', allow_pickle=True)['arr_0']
print(t5.shape)

# geometricus kmer
geo_kmer = np.load('train-val/kmer_Geometricus_Fixed.npz', allow_pickle=True)['arr_0']
print(geo_kmer.shape)
kmer_means = np.asarray([6.964, 15.231, 11.939, 16.131])
kmer_stds = np.asarray([0.559, 1.215, 0.927, 1.929])

# geometricus radius
geo_radius = np.load('train-val/radius_Geometricus_Fixed.npz', allow_pickle=True)['arr_0']
print(geo_radius.shape)
radius_means = np.asarray([6.594, 15.934, 11.919, 14.685])
radius_stds = np.asarray([0.659, 2.284, 1.402, 1.544])

# hphobicity
hphobicity = np.load('train-val/DomDet_Train-Val_Hphobicity.npz', allow_pickle=True)['arr_0']
print(hphobicity.shape)

# ss
ss = np.load('train-val/ss_proc_Train-Val.npz', allow_pickle=True)['arr_0']
print(ss.shape)

ds_size = len(ss)

# cmaps
cmaps = np.load('train-val/Dataset_ContactMaps.npz', allow_pickle=True)['arr_0']
print(cmaps.shape)

# shape: (max_res, n_features) = (3042, 1024+8+1+5+3042) = (3042, 4080)
for i in range(ds_size):
	# output_dict
	chain_dict = {}

	feature_vec = np.zeros((max_res, 4080), dtype=np.float32)
	print(feature_vec.shape)
	num_res = len(hphobicity[i])
	print(num_res)

	# add t5
	print(t5[i].shape)
	feature_vec[0:num_res, 0:1024] = t5[i]

	# add geo 
	print(geo_kmer[i].shape, geo_radius[i].shape)

	feature_vec[0:num_res, 1024:1028] = ((geo_kmer[i] - kmer_means) / kmer_stds)
	feature_vec[0:num_res, 1028:1032] = ((geo_radius[i] - radius_means) / radius_stds)

	# add hphobicity 
	h_vec = np.expand_dims(np.asarray(hphobicity[i]), axis=1)
	print(h_vec.shape)
	feature_vec[0:num_res, 1032:1033] = ((h_vec - [-0.270]) / [3.079])

	# add ss
	ss_vec = np.asarray(ss[i])
	print(ss_vec.shape)
	feature_vec[0:num_res, 1033:1038] = ss_vec

	# add cmaps
	feature_vec[0:num_res, 1038:1038+num_res] = cmaps[i]

	chain_dict['X'] = feature_vec

	# make y label
	y = np.zeros((max_res, 1), dtype=np.float32)
	for j in range(len(dom_annots[i])):
		if dom_annots[i][j] == 'N':
			y[j] = 1 # not domain
		else:
			y[j] = 2 # domain

	chain_dict['y'] = y
	chain_dict['desc'] = chain_desc[i]
	print(chain_dict)

	np.savez_compressed('processed/train-val/' + str(i) + '.npz', chain_dict)
	
'''
arr.item()['X']
'''