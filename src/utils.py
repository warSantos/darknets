import pickle
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
