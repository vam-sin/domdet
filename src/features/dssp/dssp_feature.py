from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBList
import pandas as pd 
import numpy as np 

ds = pd.read_csv('ds_final_remake_domconds_annots.csv')
print(ds)

chain_desc = list(ds["chain-desc"])
pdb_res_nums = list(ds["pdb-res-nums"])
chain_seq = list(ds["pdb-seq"])
pdb_list = []

for i in range(len(chain_desc)):
    pdb_list.append(chain_desc[i].split('|')[2])

# parse structure
p = PDBParser()

ss_list = []

for i in range(len(chain_seq)):
    sec_structure = ''
    print(i, len(pdb_list), pdb_list[i])
    chain_id = pdb_list[i][4]

    res_nums = pdb_res_nums[i].split(';')
    start = int(res_nums[0])
    stop = int(res_nums[len(res_nums)-2])
    
    structure = p.get_structure(pdb_list[i], 'chain_pdb/' + pdb_list[i] + '.pdb')
    model = structure[0]
    dssp = DSSP(model, 'chain_pdb/' + pdb_list[i] + '.pdb', dssp='mkdssp')
    
    print(list(dssp.keys()))
    print(len(list(dssp.keys())), start, stop)
    for z in range(start, stop+1):
        try:
            a_key = (chain_id, (' ', z, ' '))
            sec_structure += dssp[a_key][2]
        # print(sec_structure)
        except:
            sec_structure += '-'

    if len(sec_structure) != len(chain_seq[i]):
        for k in range(len(chain_seq[i]) - len(sec_structure)):
            sec_structure += '-'
    if len(sec_structure) != len(chain_seq[i]):
        sec_structure = sec_structure[:len(chain_seq[i])]
    if len(sec_structure) != len(chain_seq[i]):
        print(len(sec_structure), len(chain_seq[i]))
        break
    # if ('H' not in sec_structure) and ('S' not in sec_structure) and ('T' not in sec_structure):
    #     print("ONLY ----")
    #     break

    ss_list.append(sec_structure)

ds['dssp_ss'] = ss_list

ds.to_csv('ds_final_remake_domconds_annots_ss.csv')
