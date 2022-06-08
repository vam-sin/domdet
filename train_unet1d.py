import re
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, precision_recall_curve, auc, matthews_corrcoef

from UNet_1D import UNET
from data_generator import PytorchDataset
import wandb



def get_hyperparameters(path='unet_hp.csv'):
    # load hyper params from csv select one set of HPs to evaluate
    # Designed for array job grid-search over HPs
    row_index = int(sys.argv[1]) -1
    hyper_df = pd.read_csv(path)
    hp = hyper_df.loc[row_index].to_dict()
    hp = {k: v for k, v in hp.items() if not re.search('[A-Za-z]*_e\d+', k)} # remove the rows of the data frame that contain performance metrics from previous runs
    for k, v in hp.items():
        if v % 1 == 0:
            hp[k] = int(v)
    print('loaded csv hyperparams')
    hp = make_unet_channels(hp)
    return hp, row_index

def make_unet_channels(hp):
    depth = 5
    for i in range(depth):
        hp['c' + str(depth-i)] = int(hp['max_channels'] / 2**i)
    return hp

class DiceLoss:
    def __init__(self, denominator_exponent=1, smooth=0.01):
        self.denominator_exponent = denominator_exponent
        self.smooth = smooth

    def __call__(self, y_pred, y_true):
        intersection = (y_pred * y_true).sum()
        denominator = (y_pred.sum() + y_true.sum() + self.smooth)
        if self.denominator_exponent != 1:
            denominator = denominator**self.denominator_exponent
        return 1 - (intersection + self.smooth) / denominator

class WeightedCrossEntropy:
    def __init__(self, weighting):
        self.weighting = weighting

    def __call__(self, y_pred, y_true):
        loss_pos = self.weighting * y_true * torch.log(y_pred)
        loss_neg = (1 - y_true) * torch.log(1 - y_pred)
        loss = -1 * (loss_pos + loss_neg)
        return torch.sum(loss)


def select_f1_threshold(precision, recall, thresholds):
    denominator = recall + precision
    denominator[denominator==0] = 1
    f1_scores = 2 * recall * precision / denominator
    return thresholds[np.argmax(f1_scores)]


def get_scores(y_true, y_pred):
    y_true = y_true > 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    classification_threshold = select_f1_threshold(precision, recall, thresholds)
    binarized_pred = y_pred >= classification_threshold

    if binarized_pred.any():
        mcc = matthews_corrcoef(y_true, binarized_pred)
    else:
        mcc = 0
    return {
        'roc_auc': roc_auc_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, binarized_pred),
        'precision': precision_score(y_true, binarized_pred, zero_division=0),
        'recall': recall_score(y_true, binarized_pred),
        'MCC': mcc,
        'PRAUC':auc(recall, precision)
    }

def evaluate_on_test(model, test_dir = 'features/processed/test/', debug_mode=False):
    test_set = PytorchDataset(dir=test_dir, max_res=480)
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True)
    all_y = np.array([])
    all_pred = np.array([])
    b_counter = 0
    with torch.set_grad_enabled(False):
        for (x, mask), y in test_generator:
            if debug_mode and b_counter > 1:
                break
            y_pred = model(x)
            x, y, y_pred, mask = x.numpy(), y.numpy(), y_pred.numpy(), mask.numpy().astype(bool)
            all_y = np.append(all_y, y[mask])
            all_pred = np.append(all_pred, y_pred[mask])
            b_counter += 1
    scores = get_scores(all_y, all_pred)
    wandb.log(scores)
    for k, v in scores.items():
        print(k,v)
    return scores

if __name__=="__main__":
    debug_mode = False
    wandb.init(entity="cath", project="domdet")
    train_dir = 'features/processed/train-val/'
    test_dir = 'features/processed/test/'
    model_save_dir = 'unet_logs/'
    hp, row_index = get_hyperparameters(path='unet_hp.csv')
    if debug_mode:
        hp['max_channels'] = 512
    BATCH_SIZE = hp.get('batch', 4)
    model = UNET(hp)
    epochs = hp.get('epochs', 500)
    if hp.get('loss_function', 'cross_entropy') == 'cross_entropy':
        if hp.get('weighting', 1) == 1:
            loss_function = torch.nn.BCELoss()
        else:
            loss_function = WeightedCrossEntropy(hp['weighting'])
    elif hp['loss_function'] == 'dice':
        loss_function = DiceLoss(hp.get('denominator_exponent', 1))

    training_set = PytorchDataset(dir=train_dir, max_res=480)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)

    use_cuda = torch.cuda.is_available()
    print('USE CUDA:', use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('Batch Size:', BATCH_SIZE)
    model.to(device=device)
    model = torch.nn.DataParallel(model)
    wandb.watch(model, log_freq=1000)
    if 'weight_decay' in hp:
        weight_decay = hp['weight_decay']
    else:
        weight_decay = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=weight_decay)
    for e in range(epochs):
        b_counter = 0
        for (x, mask), y in training_generator:
            if debug_mode and b_counter > 1:
                break
            try:
                print(b_counter, end=' ')
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                y_pred = model(x)
                if not isinstance(loss_function, list):
                    loss = loss_function(y_pred, y)
                    wandb.log({"loss": loss})
                else:
                    loss = 0
                    for l_idx, loss_f in enumerate(loss_function):
                        one_loss = loss_f(y_pred, y)
                        wandb.log({str(loss_f).split()[1]: one_loss})
                        loss += one_loss
                if torch.isnan(loss):
                    raise ValueError('nan loss encountered')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
                optimizer.zero_grad()
                b_counter += 1
            except Exception as exe:
                print(exe)
        print('\nepochs complete:', e)
        evaluate_on_test(model, test_dir='features/processed/test/', debug_mode=debug_mode)


