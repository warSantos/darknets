import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split, TensorDataset

from src.utils import load_test, load_train_test

class DataHandler:
    
    def __init__(self,
                 DATA_SOURCE: str,
                 base_models: list,
                 day: str,
                 fold: int,
                 label_to_idx: dict,
                 intersec: set = None) -> None:
    
        self.DATA_SOURCE = DATA_SOURCE
        self.base_models = base_models
        self.day = day
        self.fold = fold
        self.label_to_idx = label_to_idx
        self.intersec = intersec
    
    def build_data_loaders(self):
        
        X_train, X_test, y_train, y_test = load_train_test(
            self.DATA_SOURCE,
            self.base_models,
            self.day,
            self.fold,
            self.label_to_idx
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
    
    def build_test_dataloader(self):
        
        X_test, y_test = load_test(
            self.DATA_SOURCE,
            self.base_models,
            self.day,
            self.fold,
            self.label_to_idx,
            self.intersec
        )
        self.y_test = y_test
        self.dim = X_test.shape[1]
        self.n_labels = len(self.label_to_idx)

        test_set = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        self.test_loader = DataLoader(test_set, batch_size=64, num_workers=4)

class NeuralNet(pl.LightningModule):
    
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
        super(NeuralNet, self).__init__()
        self.save_hyperparameters()

        # Define layers
        self.relu = nn.ReLU()
        
        self.linear_1 = nn.Linear(input_dim, int(hidden_dim / 2))
        self.dropout_1 = nn.Dropout(p=0.1)
        
        self.linear_2 = nn.Linear(int(hidden_dim / 2), hidden_dim)
        self.dropout_2 = nn.Dropout(p=0.1)
        
        self.linear_3 = nn.Linear(hidden_dim, output_dim)
        
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        
        x = self.dropout_1(self.relu(self.linear_1(x)))
        x = self.dropout_2(self.relu(self.linear_2(x)))
        return self.linear_3(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log_dict({"val_loss": loss }, prog_bar=True, on_epoch=True)
        
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        
        # Linear learning rate scheduler
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / self.trainer.max_epochs)
        return [optimizer], [scheduler]