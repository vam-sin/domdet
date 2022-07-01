import os
import numpy as np
import pickle

if __name__=="__main__":
    alphafold_dir = 'features/test_alphafold_features/'
    pae_output_dir = 'features/test_pae/'
    os.makedirs(pae_output_dir, exist_ok=True)
    for f in os.listdir(alphafold_dir):
        path = os.path.join(alphafold_dir, f)
        save_name = f.split('.')[0]
        save_path = os.path.join(pae_output_dir, save_name)
        with open(path, 'rb') as filehandle:
            feat = pickle.load(filehandle)
            pae = feat['predicted_aligned_error']
            updated_pae = np.zeros([len(pae), len(pae) + 1])
            updated_pae[:, 1:] = pae
            updated_pae[:,0] = 1
            np.save(save_path, updated_pae)
            bp=True