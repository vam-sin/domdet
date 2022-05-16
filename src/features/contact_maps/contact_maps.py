import pandas as pd  
import Bio.PDB
import numpy as np
import sys

ds = pd.read_csv('ds_final.csv')
chain_desc = list(ds["chain-desc"])
pdb_list = []

for i in range(len(chain_desc)):
    pdb_list.append(chain_desc[i].split('|')[2])

# print(pdb_list[5396])

def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    try:
    	diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    	dist = np.sqrt(np.sum(diff_vector * diff_vector))
    except:
    	dist = 20.0
    return dist 

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

maps = []

for i in range(len(pdb_list)):
	pdb_code = pdb_list[i]
	pdb_filename = "chain_pdb/" + pdb_list[i]
	chain_id = str(pdb_code[len(pdb_code)-1])
	if chain_id == '0' and pdb_code != '4bpo0':
		chain_id = ' '
	print(i, len(pdb_list), pdb_list[i])
	structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
	model = structure[0]
	# for i in model:
	# 	print(i)
	print(model[chain_id])
	dist_matrix = calc_dist_matrix(model[chain_id], model[chain_id])
	contact_map = np.asarray((dist_matrix < 5.0).astype(int))
	# print(contact_map)
	maps.append(contact_map)

maps = np.asarray(maps)
np.set_printoptions(threshold=sys.maxsize)
print(maps[0])
np.savez_compressed('Dataset_ContactMaps.npz', maps)