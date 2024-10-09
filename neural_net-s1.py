
import os
import json
import pickle
from glob import glob
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils import under_sampling
from src.neural_net import NeuralNet


def get_X_y(DATA_SOURCE: str,
            model: str,
            day: str,
            train_test: str,
            fold: int,
            label_to_idx: dict,
            ip_set: list):
    
    df = pd.read_csv(f"data/2022/input/reps/out/k3/{DATA_SOURCE}/{train_test}/{model}_{day}_fold0{fold}.csv")
    
    # Taking the intersection.
    df = df[df.src_ip.isin(ip_set)].sort_values(by=["src_ip"])
    df_copy = df.drop(columns=["src_ip"])
    
    y = df_copy.label.values
    X = df_copy.drop(columns=["label"]).values
    
    
    # applying undersampling.
    if train_test == "train":
        print(pd.Series(y).value_counts())
        print(f"{DATA_SOURCE} BEFORE UND.: ", X.shape, y.shape)
        X, y = under_sampling(X, y)
        print(f"{DATA_SOURCE} AFTER UND.: ", X.shape, y.shape)
    
    l = [ label_to_idx[i] for i in y]
    return X, l

def load_train_test(sources: list,
                    models: list,
                    day: str,
                    fold: int,
                    label_to_idx: dict,
                    inter_set_train: list,
                    inter_set_test: list):
    
    X_train, X_test = [], []
    for source, model in product(sources, models):
        
        X, y_train = get_X_y(source, model, day, "train", fold, label_to_idx, inter_set_train)
        X_train.append(X)
        
        X, y_test = get_X_y(source, model, day, "test", fold, label_to_idx, inter_set_test)
        X_test.append(X)
    
    X_train = np.hstack(X_train)
    X_test = np.hstack(X_test)
    
    return X_train, X_test, y_train, y_test
    
class DataHandler:
    
    def __init__(self,
                 sources: str,
                 base_models: list,
                 day: str,
                 fold: int,
                 label_to_idx: dict,
                 inter_set_train: list,
                 inter_set_test: list) -> None:
    
        self.sources = sources
        self.base_models = base_models
        self.day = day
        self.fold = fold
        self.label_to_idx = label_to_idx
        self.inter_set_train = inter_set_train
        self.inter_set_test = inter_set_test
    
    def build_data_loaders(self) -> None:
        
        X_train, X_test, y_train, y_test = load_train_test(
            self.sources,
            self.base_models,
            self.day,
            self.fold,
            self.label_to_idx,
            self.inter_set_train,
            self.inter_set_test
        )
        
        self.y_test = y_test
        self.dim = X_train.shape[1]
        self.n_labels = len(self.label_to_idx)

        train_set = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_set = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

        # Split the dataset into training and validation sets
        train_len = int(0.9 * len(train_set))
        val_len = len(train_set) - train_len

        train_set, val_set = random_split(train_set, [train_len, val_len])

        # Create DataLoaders
        self.train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_set, batch_size=64, num_workers=4)
        self.test_loader = DataLoader(test_set, batch_size=64, num_workers=4)

def get_intersec(strat: dict, day: str, sources: list, fold: int) -> tuple:
    
    train_ips = set(strat[day][sources[0]][fold][0]).intersection(set(strat[day][sources[1]][fold][0]))
    test_ips = set(strat[day][sources[0]][fold][1]).intersection(set(strat[day][sources[1]][fold][1]))
    return train_ips, test_ips
    
if __name__=="__main__":
    
    SOURCES = ["darknet", "honeypot"]
    BASE_MODELS = ["features", "idarkvec", "igcngru_features"]

    N_FOLDS = 10
    MAX_EPOCHS = 15

    probs_cols = [
        "censys",
        "driftnet",
        "internetcensus",
        "intrinsec",
        "ipip",
        "mirai",
        "onyphe",
        "rapid7",
        "securitytrails",
        "shadowserver",
        "shodan",
        "u_mich",
        "unk_bruteforcer",
        "unk_exploiter",
        "unk_spammer",
        "unknown"
    ]
    probs_cols.sort()

    label_to_idx = {l: idx for idx, l in enumerate(probs_cols)}

    days = sorted([ f.split('/')[-1].split('_')[-2] for f in glob(f"data/2022/input/reps/out/k3/darknet/test/idarkvec*_fold00.csv") ])

    with open("data/2022/input/skf/stratification/stratification.json", 'r') as fd:
        splits = json.load(fd)

    predictions = {}
    #for day, fold in product(days, np.arange(N_FOLDS)):
    for day, fold in product(["20221022"], [0]):
        
        # Getting intersection sets.
        train_ips, test_ips = get_intersec(splits, day, SOURCES, fold)
        
        # Loading data.
        data = DataHandler(SOURCES, BASE_MODELS, day, fold, label_to_idx, train_ips, test_ips)
        data.build_data_loaders()
        
        # Setting network's size
        input_dim = data.dim
        hidden_dim = input_dim
        output_dim = data.n_labels
        
        # Model Logger.
        output = f"data/2022/output/nn/1/{data.day}/{data.fold}/"
        os.makedirs(output, exist_ok=True)

        checkpoint_callbacker = ModelCheckpoint(
            monitor="val_loss",
            dirpath=output,
            filename="model",
            save_top_k=1,
            mode="min"
        )

        # Define model, trainer, and train the model
        model = NeuralNet(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, lr=1e-3
        )
        trainer = pl.Trainer(max_epochs=MAX_EPOCHS, callbacks=[checkpoint_callbacker])
        trainer.fit(model, data.train_loader, data.val_loader)
        
        # Load the best checkpoint
        best_model_path = checkpoint_callbacker.best_model_path
        best_model = NeuralNet.load_from_checkpoint(best_model_path)

        # Prediction
        y_hat = trainer.predict(best_model, dataloaders=data.test_loader)
        y_hat = (
            torch.cat(y_hat).cpu().numpy()
        )  # Combine y_hat and move to CPU

        print(F"DAY: {day} / FOLD: {fold} - M-F1: {(100 * f1_score(data.y_test, y_hat, average='macro')):.2f}")
        if day not in predictions:
            predictions[day] = {}
        predictions[day][fold] = {
            "y_test": data.y_test,
            "y_hat": y_hat
        }

    output = f"data/2022/output/nn/1"
    with open(f"{output}/preds.pkl", "wb") as fd:
        pickle.dump(predictions, fd)