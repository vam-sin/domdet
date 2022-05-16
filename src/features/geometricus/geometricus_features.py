import pandas as pd 
import numpy as np 
import prody
from geometricus import MomentInvariants, SplitType

ds = pd.read_csv('ds_final.csv')

chain_desc = list(ds["chain-desc"])
pdb_list = []

for i in range(len(chain_desc)):
    pdb_list.append(chain_desc[i].split('|')[2])
pdbs = []
for i in range(len(pdb_list)):
	print(pdb_list[i])
	if i > 0 and i % 50 == 0:
		print(i)
	if i >=10:
		break
	pdb_code = pdb_list[i]
	pdb_filename = "whole_pdb/" + pdb_list[i][:4]
	chain_id = str(pdb_code[len(pdb_code)-1])
	if chain_id == '0' and pdb_code != '4bpo0':
		chain_id = ' '
	print(pdb_filename, chain_id, pdb_code)
	# pdbs.append(prody.parsePDB(pdb_filename, chain=chain_id))

# invariants_kmer = []
# invariants_radius = []

# for i in range(len(pdb_list)):
# 	if i > 0 and i % 50 == 0:
# 		print(i)
# 	invariants_kmer.append(MomentInvariants.from_prody_atomgroup(pdb_list[i], pdbs[i], split_type=SplitType.KMER, split_size=16))
# 	invariants_radius.append(MomentInvariants.from_prody_atomgroup(pdb_list[i], pdbs[i], split_type=SplitType.RADIUS, split_size=10))

# invariants_kmer = np.asarray(invariants_kmer)
# print(invariants_kmer.shape)

# invariants_radius = np.asarray(invariants_radius)
# print(invariants_radius.shape)

# np.savez_compressed('kmer_Geometricus.npz', invariants_kmer)
# np.savez_compressed('radius_Geometricus.npz', invariants_radius)