import re
import sys
import torch
import pandas as pd

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
    return hp, row_index

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

if __name__=="__main__":
    debug_mode = True
    wandb.init(entity="cath", project="testing")
    train_dir = 'features/processed/train-val/'
    test_dir = 'features/processed/test/'
    model_save_dir = 'unet_logs/'
    hp, row_index = get_hyperparameters(path='unet_hp.csv')
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
