
import os
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
        ["2", "darknet"],
        ["3", "honeypot"]
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

    predictions = {}
    for case_id, DATA_SOURCE in CASES:
        for day, fold in product(days, np.arange(N_FOLDS)):
            
            # Loading data.
            data = DataHandler(DATA_SOURCE, BASE_MODELS, day, fold, label_to_idx)
            data.build_data_loaders()
            
            # Setting network's size
            input_dim = data.dim
            hidden_dim = input_dim
            output_dim = data.n_labels
            
            # Model Logger.
            output = f"data/2022/output/nn/{case_id}/{data.day}/{data.fold}/"
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

            print(F"{DATA_SOURCE} / {day} / FOLD: {fold} - M-F1: {(100 * f1_score(data.y_test, y_hat, average='macro')):.2f}")
            if day not in predictions:
                predictions[day] = {}
            predictions[day][fold] = {
                "y_test": data.y_test,
                "y_hat": y_hat
            }

        output = f"data/2022/output/nn/{case_id}"
        with open(f"{output}/preds.pkl", "wb") as fd:
            pickle.dump(predictions, fd)