import os
import requests
import json
import numpy as np
import pandas as pd
from Bio import pairwise2

def add_match_status_column(pae_matrix, match_status):
    new_matrix = np.zeros([len(pae_matrix), len(pae_matrix) + 1])
    new_matrix[:,1:] = pae_matrix
    new_matrix[:,0] = match_status
    return new_matrix

# def save_pae_np(pae, dir, pdbid, start_res, end_res, match_status):
#     """
#     saves the PAE as a matrix where each row should be interpreted as
#     the PAE between residue of row_index i, and each of it's neighbours (for each column j)
#     """
#     n_residues = (end_res +1) - start_res
#     res1_in_range = np.logical_and(pae['residue1'] >= start_res+1, pae['residue1']<= end_res+1)
#     res2_in_range = np.logical_and(pae['residue2'] >= start_res+1, pae['residue2']<= end_res+1)
#     selection_index = np.where(np.logical_and(res1_in_range, res2_in_range))
#     distances = np.array(pae['distance'])[selection_index]
#     pae_array = distances.reshape([n_residues, n_residues])
#     pae_array = add_match_status_column(pae_array, match_status)
#     save_path = os.path.join(dir, pdbid)
#     np.save(save_path, pae_array)

def save_pae_np(pae, dir, pdbid, mapping, match_status):
    """
    saves the PAE as a matrix where each row should be interpreted as
    the PAE between residue of row_index i, and each of it's neighbours (for each column j)
    """
    n_residues = len(mapping.keys())
    pae_index = list(mapping.values())
    res1_select = [pae['residue1'][i]-1 in pae_index for i in range(len(pae['residue1']))]
    res2_select = [pae['residue2'][i]-1 in pae_index for i in range(len(pae['residue2']))]
    selection_index = np.where(np.logical_and(res1_select, res2_select))
    distances = np.array(pae['distance'])[selection_index]
    pae_array = distances.reshape([n_residues, n_residues])
    pae_array = add_match_status_column(pae_array, match_status)
    save_path = os.path.join(dir, pdbid)
    np.save(save_path, pae_array)

def get_alignment(up_seq, pdb_seq):
    alignment = pairwise2.align.globalxs(up_seq, pdb_seq, -1, -1)[0]._asdict()
    return alignment

# def get_match_status(alignment):
#     return np.array([alignment['seqA'][pdb_start:pdb_end + 1][i] == alignment['seqB'][pdb_start:pdb_end + 1][i] for i in
#               range(pdb_end + 1 - pdb_start)]).astype(int)

def get_match_status(alignment, mapping):
    pdb_seq = alignment['seqB']
    unip_seq = alignment['seqA']
    trim_pdb_seq = pdb_seq.replace('-', '')
    trim_unip_seq = unip_seq.replace('-', '')
    pdb_select = np.array(list(trim_pdb_seq))[list(mapping.keys())]
    unip_select = np.array(list(trim_unip_seq))[list(mapping.values())]
    assert len(pdb_select) == len(unip_select)
    match_status = np.array([pdb_select[i] == unip_select[i] for i in range(len(pdb_select))]).astype(int)
    return match_status

def create_mapping_dict(alignment):
    up_seq = alignment['seqA']
    pdb_seq = alignment['seqB']
    mapping = {}
    pdb_counter = 0
    up_counter = 0
    trim_pdb_seq = pdb_seq.replace('-', '')
    for i, align_pdb in enumerate(pdb_seq):
        if align_pdb != '-':
            while up_seq[up_counter] == '-':
                up_counter += 1
            mapping[pdb_counter] = up_counter
            pdb_counter +=1
            up_counter += 1
        elif up_seq[up_counter] != '-':
            up_counter += 1
    return mapping

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
if __name__=='__main__':
    df_path = 'ds_final_imp.csv'
    pae_save_dir = 'features/paes2/'
    df = pd.read_csv(df_path)
    os.makedirs(pae_save_dir, exist_ok=True)
    pdb_list = get_pdbid_list(df_path)
    completed = [f.split('.')[0] for f in os.listdir(pae_save_dir)]
    map_df = pd.read_csv('pdb_chain_uniprot.csv', header=1)
    for i, row in df.iterrows():
        try:
            pdbid = row['chain-desc'].split('|')[-1][:5]
            chain = pdbid[-1]
            if pdbid in completed:
                continue
            try:
                one_chain_map = map_df[(map_df.PDB == pdbid[:4].lower())&(map_df.CHAIN==chain)].iloc[0]
            except:
                print(pdbid + chain)
                continue
            n_residues = len(row['pdb-seq'])
            pdb_seq = row['pdb-seq']
            up_id = one_chain_map['SP_PRIMARY']
            try:
                pae_graph = get_pae(uniprot_id=up_id)
            except:
                result = get_uniprot_metadata(pdbid[:-1])
                up_keys = list(result[pdbid[:-1]]['UniProt'].keys())
                for up_id in up_keys:
                    try:
                        one_result = result[pdbid[:-1]]['UniProt'][up_id]
                        mappings = [d for d in one_result['mappings'] if d['chain_id'] == chain]
                        pae_graph = get_pae(uniprot_id=up_id)
                        break
                    except:
                        pass
            fasta = requests.get(f'https://www.uniprot.org/uniprot/{up_id}.fasta').text
            up_sequence = ''.join(fasta.split('\n')[1:])
            alignment = get_alignment(up_sequence, pdb_seq)
            # mapping = dict(zip(np.where(np.array(list(alignment['seqA'])) != '-')[0], np.where(np.array(list(alignment['seqB'])) != '-')[0]))
            mapping = create_mapping_dict(alignment)
            pae_start = min(mapping.values())
            pae_end = max(mapping.values())
            pdb_start = min(np.where(np.array(list(alignment['seqB'])) != '-')[0])
            pdb_end = max(np.where(np.array(list(alignment['seqB'])) != '-')[0])
            aligned_length  = (pae_end + 1) - pae_start
            if aligned_length - n_residues > 0.6 * n_residues:
                continue
            # per_residue_match_status = (np.array(list(alignment['seqB'][pae_start:pae_end+1])) != '-').astype('int')
            per_residue_match_status = get_match_status(alignment, mapping)
            try:
                save_pae_np(pae_graph, pae_save_dir, pdbid, mapping, per_residue_match_status)
            except:
                save_pae_np(pae_graph, pae_save_dir, pdbid, mapping, per_residue_match_status)
            bp=True
        except:
            print(row['chain-desc'])