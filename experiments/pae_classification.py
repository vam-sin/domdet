import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, f1_score, precision_score, recall_score
os.chdir('..')

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


if __name__=="__main__":
    pae_dir = 'features/paes/'
    pdb_to_label_path = make_pdb_to_label('ds_final_imp.csv')
    max_res = 350
    x,y = make_initial_x_y(max_res=max_res)
    res_start = 0
    for fname in os.listdir(pae_dir):
        try:
            pae = np.load(os.path.join(pae_dir, fname))
            n_res = min(max_res, len(pae))
            res_end = res_start + n_res
            if res_end > len(x):
                x = x[:res_start, :]
                y = y[:res_start]
                break
            label_index =pdb_to_label_path[fname.split('.')[0]]
            label = np.load(f'features/processed/train-val/{label_index}.npz', allow_pickle=True)['arr_0'].item()['y']
            label = label[:n_res, 0]
            y[res_start:res_end] = label
            x[res_start:res_end, :n_res + 1] = pae[:n_res,:max_res+1]
            res_start = res_end
        except:
            pass
    y = y-1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 13)
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
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=6)
    preds = model.predict_proba(x_test)[:,1]
    scores = get_scores(y_test, preds, threshold=0.5)
    for k,v in scores.items():
        print(k,v)
    bp=True

