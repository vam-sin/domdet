import os
import requests
import json
import numpy as np
import pandas as pd
from Bio import pairwise2

def get_alignment(alpha_seq, pdb_seq):
    alignment = pairwise2.align.globalxx(alpha_seq, pdb_seq)[0]._asdict()
    return alignment

def create_mapping_dict(alignment):
    mapping = {}
    pdb_counter = 0
    alpha_counter = 0
    for i, align_pdb in enumerate(alignment['seqB']):
        align_alpha = alignment['seqA'][i]
        if align_pdb != '-':
            if align_pdb == align_alpha: # this should always be true when using an alignment tool that can only add gaps
                mapping[pdb_keys[pdb_counter]] = alpha_counter
            pdb_counter +=1
        if align_alpha != '-':
            alpha_counter += 1
    if all([letter2name[alignment['seqA'][mapping[k]]]==v for k,v in combined.items()]):
        return mapping
    else:
        return None

def get_pdbid_list(df_path):
    df = pd.read_csv(df_path)
    return df['chain-desc'].apply(lambda x: x.split('|')[-1][:5]).tolist()

def get_uniprot_metadata(pdbid):
    url = f'https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdbid}'
    r = requests.get(url)
    result_json = r.json()
    return result_json

def get_pae(uniprot_id):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-predicted_aligned_error_v2.json"
    r = requests.get(url)
    pae_graph = r.json()[0]
    return pae_graph

def trim_pae():
    pass

df_path = 'ds_final_imp.csv'
pae_save_dir = 'features/paes2/'
df = pd.read_csv(df_path)
os.makedirs(pae_save_dir, exist_ok=True)
pdb_list = get_pdbid_list(df_path)
completed = []
map_df = pd.read_csv('pdb_chain_uniprot.csv', header=1)
for i, row in df.iterrows():
    pdbid = row['chain-desc'].split('|')[-1][:5]
    chain = pdbid[-1]
    try:
        one_chain_map = map_df[(map_df.PDB == pdbid[:4].lower())&(map_df.CHAIN==chain)].iloc[0]

    except:
        print(pdbid + chain)
        continue
    n_residues = len(row['pdb-seq'])
    try:
        assert one_chain_map.RES_END == n_residues
    except:
        label = np.load(f'features/processed/train-val/{i}.npz', allow_pickle=True)['arr_0'].item()['y']
        bp = True
    continue
    if pdbid in completed:
        continue
    try:

        result = get_uniprot_metadata(pdbid[:-1])
        up_keys = list(result[pdbid]['UniProt'].keys())
        for up_id in up_keys:
            one_result = result[pdbid]['UniProt'][up_id]
            mappings = [d for d in one_result['mappings'] if d['chain_id'] == chain]
            up_start = mappings[0]['unp_start']
            up_end = mappings[0]['unp_end']
            pae_graph = get_pae(uniprot_id=up_id)

    except:
        pass