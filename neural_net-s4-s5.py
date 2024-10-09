import os
import json
import pickle
from glob import glob
from itertools import product

import numpy as np
from sklearn.metrics import f1_score

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.neural_net import DataHandler, NeuralNet
    

if __name__=="__main__":

    BASE_MODELS = ["features", "idarkvec", "igcngru_features"]
    CASES = [
        ["4", "2", "darknet"],
        ["5", "3", "honeypot"]
    ]

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

    N_FOLDS = 10
    MAX_EPOCHS = 30

    with open("data/2022/input/skf/stratification/stratification.json", 'r') as fd:
        splits = json.load(fd)
    
    predictions = {}
    for case_id, case_ref, DATA_SOURCE in CASES:
        for day, fold in product(days, np.arange(N_FOLDS)):
            
            
            intersec = set(splits[day]["darknet"][fold][1]).intersection(set(splits[day]["honeypot"][fold][1]))
            
            # Loading data.
            data = DataHandler(DATA_SOURCE, BASE_MODELS, day, fold, label_to_idx, intersec)
            data.build_test_dataloader()

            # Load the best checkpoint
            best_model_path = f"data/2022/output/nn/{case_ref}/{data.day}/{data.fold}/model.ckpt"
            best_model = NeuralNet.load_from_checkpoint(best_model_path)

            # Prediction
            trainer = pl.Trainer()
            y_hat = trainer.predict(best_model, dataloaders=data.test_loader)
            y_hat = (
                torch.cat(y_hat).cpu().numpy()
            )  # Combine y_hat and move to CPU

            print(F"{DATA_SOURCE} / {day} / FOLD: {fold} - M-F1: {(100 * f1_score(data.y_test, y_hat, average='macro')):.2f}")
            if day not in predictions:
                predictions[day] = {}
            predictions[day][fold] = {
                "y_test": data.y_test,
                "y_hat": y_hat
            }

        output = f"data/2022/output/nn/{case_id}"
        os.makedirs(output, exist_ok=True)
        with open(f"{output}/preds.pkl", "wb") as fd:
            pickle.dump(predictions, fd)