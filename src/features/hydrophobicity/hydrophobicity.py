import pandas as pd  
import numpy as np 

# kd scale:
res_h = {'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5, 'X': 0, 'U': 2.5, 'Z': -3.5, 'B': -3.5}

print(len(res_h.keys()))

ds = pd.read_csv('ds_final_remake_domconds_annots_ss.csv')

seq = list(ds["pdb-seq"])

hydro = []

for i in range(len(seq)):
	prot_h = []
	for j in range(len(seq[i])):
		prot_h.append(res_h[seq[i][j]])
	hydro.append(prot_h)

hydro = np.asarray(hydro)
print(hydro.shape)
np.savez_compressed('DomDet_Train-Val_Hphobicity.npz', hydro)