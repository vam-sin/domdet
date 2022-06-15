import os
import requests
import json
import numpy as np
import pandas as pd



url2 = "https://alphafold.ebi.ac.uk/files/AF-P03023-F1-predicted_aligned_error_v2.json"

def get_pdbid_list(df_path):
    df = pd.read_csv(df_path)
    return df['chain-desc'].apply(lambda x: x.split('|')[-1][:4]).tolist()

def save_pae(pae, dir, pdbid):
    save_path = os.path.join(dir, pdbid) + '.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(pae, f, ensure_ascii=False)
        # json.dump(pae, f, ensure_ascii=False, indent=4)

def save_pae_np(pae, dir, pdbid):
    n_residues = max(pae['residue1'])
    pae_array = np.array(pae['distance']).reshape([n_residues, n_residues])
    save_path = os.path.join(dir, pdbid)
    np.save(save_path, pae_array)


def get_pae(uniprot_id):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-predicted_aligned_error_v2.json"
    r = requests.get(url)
    pae_graph = r.json()[0]
    return pae_graph

def get_uniprot_from_pdb(pdbid):
    url = f'https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdbid}'
    r = requests.get(url)
    result_json = r.json()
    return list(result_json[pdbid]['UniProt'].keys())

pae_save_dir = 'features/paes/'
os.makedirs(pae_save_dir, exist_ok=True)
# pdbid_list = [f.split('.')[0] for f in os.listdir('../PPI_site_predictor/pdb_files/')]
pdbid_list = get_pdbid_list('ds_final.csv')
completed = [f.split('.')[0] for f in os.listdir(pae_save_dir)]
for pdbid in pdbid_list:
    if pdbid in completed:
        continue
    try:
        up_ids = get_uniprot_from_pdb(pdbid)
    except Exception as exc:
        print(f'error getting uniprot for {pdbid}\n', exc)
        continue
    for up_id in up_ids:
        try:
            pae_graph = get_pae(uniprot_id=up_id)
            save_pae_np(pae_graph, pae_save_dir, pdbid)
            break
        except:
            pass


