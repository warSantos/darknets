import pickle
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import pickle


def save_pickle(data, file_path):
    
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(file_path):

    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def get_classifier(classifier):
    
    if classifier == "lr":
        return LogisticRegression(n_jobs=-1, max_iter=500)
    elif classifier == "rf":
        return RandomForestClassifier(n_jobs=-1, max_depth=8, n_estimators=200)
    else:
        raise(f"Option: {classifier} not valid.")

def under_sampling(X: np.ndarray, y: np.ndarray) -> tuple:
    
    # applying undersampling.
    n = pd.Series(y).value_counts().values[2]
    us = RandomUnderSampler(sampling_strategy={"unknown": n, "mirai": n}, random_state=42)
    us_idxs, y = us.fit_resample(np.arange(X.shape[0]).reshape(-1, 1), y)
    us_idxs = us_idxs.reshape(-1)
    X = X[us_idxs]
    return X, y

def get_X_y(DATA_SOURCE: str,
            model: str,
            day: str,
            train_test: str,
            fold: int,
            label_to_idx: dict,
            intersec: set = None):
    
    df = pd.read_csv(f"data/2022/input/reps/out/k3/{DATA_SOURCE}/{train_test}/{model}_{day}_fold0{fold}.csv")
    
    # Applying intersection if it's needed.
    if intersec is not None:
        df = df[df.src_ip.isin(intersec)]
    
    df_copy = df.drop(columns=["src_ip"])
    y = df_copy.label.values
    X = df_copy.drop(columns=["label"]).values
    
    # applying undersampling.
    if train_test == "train":
        X, y = under_sampling(X, y)
    return X, [ label_to_idx[i] for i in y]


def load_test(DATA_SOURCE: str,
              models: list,
              day: str,
              fold: int,
              label_to_idx: dict,
              intersec: set = None):
    
    X_test = []
    for model in models:
        
        X, y_test = get_X_y(DATA_SOURCE, model, day, "test", fold, label_to_idx, intersec)
        X_test.append(X)
        
    X_test = np.hstack(X_test)
    
    return X_test, y_test

def load_train_test(DATA_SOURCE: str,
                    models: list,
                    day: str,
                    fold: int,
                    label_to_idx: dict):
    
    X_train, X_test = [], []
    for model in models:
        
        X, y_train = get_X_y(DATA_SOURCE, model, day, "train", fold, label_to_idx)
        X_train.append(X)
        
        X, y_test = get_X_y(DATA_SOURCE, model, day, "test", fold, label_to_idx)
        X_test.append(X)
    
    X_train = np.hstack(X_train)
    X_test = np.hstack(X_test)
    
    return X_train, X_test, y_train, y_test