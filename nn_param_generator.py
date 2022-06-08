import pandas as pd

conv_hyperparams = dict(
            batch_size= [16],
            k_size= [5, 7, 11],
            filters=[32, 64, 128],
            dense_layers= [1,2,3,4],
            learning_rate= [1e-5, 1e-4, 1e-3, 1e-6],
            max_res=[300],
            n_features= [4080],
            conv_layers= [2,3,5],
            epochs= [200],

)

hyperparams = dict(
    out_1 = [10, 16, 32],
    k_size = [4, 16],
    dropout=[0],
    out_2=[10, 16],
    chout=[100],
    ff_layers = [2,3],
    lr=[0.001]
)

alex_hyperparams = dict(
    chan = [64, 32, 128, 16],
    lr=[0.1, 0.001, 0.0001, 0.00002],
    batch = [4]
)

kalasanty_hyperparams = dict(
    lr=[1e-6],
    weighting=[5, 50],
    batch = [4],
    k_size = [3, 7, 11],
)

unet_hp = dict(
    lr=[1e-5, 1e-6, 1e-7, 1e-8],
    weighting=[5, 50],
    batch = [4],
    k_size = [3, 7, 11],
    max_channels=[4096, 2048, 1024]
)

def calc_n_rows(hp):
    combinations = 1
    for n in [len(v) for v in hp.values()]:
        combinations *= n
    return combinations

def make_divisor_lookup(hyperparams):
    divisor_dict = {}
    for i, k in enumerate(hyperparams.keys()):
        keys_after = list(hyperparams.items())[i+1:]
        n_divisor = 1
        for n_values in [len(item[1]) for item in keys_after]:
            n_divisor *= n_values
        divisor_dict[k] = n_divisor
    return divisor_dict

def get_val(k, i, divisor_dict, hyperparams):
    round = i // divisor_dict[k]
    index = round % len(hyperparams[k])
    return hyperparams[k][index]

def make_param_df(hyperparams):
    df_rows = []
    n_rows = calc_n_rows(hyperparams)
    divisor_dict = make_divisor_lookup(hyperparams)
    for i in range(n_rows):
        new_row = {k:get_val(k, i, divisor_dict, hyperparams) for k in hyperparams.keys()}
        df_rows.append(new_row)
    return df_rows


df_rows = make_param_df(unet_hp)
df = pd.DataFrame(df_rows)
df = df.sample(frac=1.0)
df.to_csv('unet_hp.csv', index=False)
