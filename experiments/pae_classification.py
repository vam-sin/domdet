import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
os.chdir('..')

"""
Copy alphafold features from the cluster:
rsync -av jwells@pchuckle.cs.ucl.ac.uk:/home/jwells/alphafold/ClusterColabFold/ppi_representation_pickles/ .


Copy MSAs to the cluster
cd /Users/judewells/Documents/dataScienceProgramming/alphafold_funsite/domdet 

rsync -av saved_msa_ppi jwells@pchuckle.cs.ucl.ac.uk:/home/jwells/alphafold/ClusterColabFold/
"""

def make_pdb_to_label(df_path):
    df = pd.read_csv(df_path)
    return dict(zip(df['chain-desc'].apply(lambda x: x.split('|')[-1]), df.index.tolist()))

def make_initial_x_y(n_samples=2000*150, max_res=350):
    x = np.zeros([n_samples, max_res +1])
    y = np.zeros(n_samples)
    return x,y

def get_scores(y_true, y_pred, threshold=0.5):
    binarized_pred = y_pred >= threshold
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return {
        'roc_auc': roc_auc_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, binarized_pred),
        'precision': precision_score(y_true, binarized_pred),
        'recall': recall_score(y_true, binarized_pred),
        'MCC': matthews_corrcoef(y_true, binarized_pred),
        'f1': f1_score(y_true, binarized_pred),
        'PRAUC':auc(recall, precision)
    }

def make_test_set(dir, test_df_path, max_res):
    """
    This function takes the test dataset and also the directory where the PAEs are stored
    and returns a processed dataset of the PAE features and their corresponding labels
    :param dir:
    :param test_df:
    :return:
    """

    df = pd.read_csv(test_df_path)
    total_residues = df['Chain-Sequence'].apply(lambda x: len(x)).sum()
    x_test = np.zeros([total_residues, max_res+1])
    y_test = np.zeros(total_residues)
    res_start = 0
    for i, row in df.iterrows():
        fname = row['PDB-ID'] + 'A' + '.npy'
        try:
            pae = np.load(os.path.join(dir, fname))
        except FileNotFoundError:
            continue
        assert set(np.array(list(row['chain-domain-annots']))) == {'N', 'D'}
        one_chain_y = np.where(np.array(list(row['chain-domain-annots']))=='D', 1,0)
        res_end = res_start + len(one_chain_y)
        n_col = min(max_res, len(one_chain_y)) +1
        y_test[res_start:res_end] = one_chain_y
        x_test[res_start:res_end, :n_col] = pae[:, :n_col]
        res_start += len(one_chain_y)

    return x_test[:res_end], y_test[:res_end]

def make_alphafold_features(pdbid, alpha_dir='alpha_dom_pickles/'):
    with open(alpha_dir + pdbid, 'rb') as filehandle:
        feat = pickle.load(filehandle)
        bp=True
    pass

def make_train_set(pae_dir, max_res):
    x, y = make_initial_x_y(max_res=max_res)
    res_start = 0
    for fname in os.listdir(pae_dir):
        try:
            pae = np.load(os.path.join(pae_dir, fname))
            n_res = min(max_res, len(pae))
            res_end = res_start + n_res
            if res_end > len(x):# if we reach the end of the max dataset size then trim to last completed chain
                x = x[:res_start, :]
                y = y[:res_start]
                break
            pdbid = fname.split('.')[0]
            pdb_name_key = pdbid[:-1].lower() + pdbid[-1].upper()
            label_index =pdb_to_label_path[pdb_name_key]
            label = np.load(f'features/processed/train-val/{label_index}.npz', allow_pickle=True)['arr_0'].item()['y']
            label = label[:n_res, 0]
            try:
                assert 0 not in set(label)
            except:
                print(pdbid, 'non matching sequence length')
                bp=True
                continue
            y[res_start:res_end] = label
            x[res_start:res_end, :n_res + 1] = pae[:n_res,:max_res+1]
            res_start = res_end
        except:
            pass
    non_zero_indices = np.where(x.sum(axis=1)!=0)[0]
    return x[non_zero_indices], y[non_zero_indices]



if __name__=="__main__":
        pae_dir = 'features/train_colabfold_paes/'
        test_pae_dir = 'features/test_pae/'
        pdb_to_label_path = make_pdb_to_label('ds_final_imp.csv')
        max_res = 350
        x_test, y_test = make_test_set(test_pae_dir, 'casp_test_final.csv', max_res)
        x_test5, y_test5 = make_test_set('features/r5_test_pae/', 'casp_test_final.csv', max_res)
        x, y = make_train_set(pae_dir, max_res)
        y = y-1
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 13)
        model  = XGBClassifier(
            n_estimators=500,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=7,
            gamma=1,
            reg_apha=1,
            objective='binary:logistic',
            scale_pos_weight=0.05,
            njobs=-1,
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=6)
        preds = model.predict_proba(x_val)[:,1]
        scores = get_scores(y_val, preds, threshold=0.5)
        print('---validation scores---')
        for k,v in scores.items():
            print(k,round(v, 4))
        bp=True
        print('---test scores---')
        test_preds = model.predict_proba(x_test)[:,1]
        scores = get_scores(y_test, test_preds, threshold=0.5)
        for k,v in scores.items():
            print(k,round(v, 4))
        print('---r5 test scores---')
        test_preds5 = model.predict_proba(x_test5)[:,1]
        scores2 = get_scores(y_test5, test_preds5, threshold=0.5)
        for k,v in scores2.items():
            print(k,round(v, 4))



