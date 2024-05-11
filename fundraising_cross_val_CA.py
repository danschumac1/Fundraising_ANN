import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import argparse
import json

def setup_args():
    parser = argparse.ArgumentParser(description='Train a neural network for donor prediction.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    parser.add_argument('--layers', nargs='+', type=int, default=[100, 200], help='List of sizes for each hidden layer.')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds for cross-validation.')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate.')
    parser.add_argument('--i', type=int, required=True, help='what expirement number?')
    return parser.parse_args()

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, index_col='Unnamed: 0')
    df['log_home_value'] = np.log(df['home_value'] + 1)
    df['log_lifetime_gifts'] = np.log(df['lifetime_gifts'] + 1)
    df['log_largest_gift'] = np.log(df['largest_gift'] + 1)
    df['log_avg_gift'] = np.log(df['avg_gift'] + 1)
    df.drop(['home_value', 'lifetime_gifts', 'largest_gift', 'avg_gift'], axis=1, inplace=True)

    df['zipconvert'] = df[['zipconvert2', 'zipconvert3', 'zipconvert4', 'zipconvert5']].apply(
        lambda row: 'zc' + str(row.idxmax()[-1]) if pd.notna(row.idxmax()) else 'unknown', axis=1
    )
    test = df[df['type'] == 'test']
    df = df[df['type'] != 'test']
    df.drop(['zipconvert2', 'zipconvert3', 'zipconvert4', 'zipconvert5', 'type'], axis=1, inplace=True)
    test.drop(['zipconvert2', 'zipconvert3', 'zipconvert4', 'zipconvert5', 'type'], axis=1, inplace=True)

    cat_cols = ['homeowner', 'female', 'zipconvert', 'wealth', 'income', 'num_child']
    cont_cols = [col for col in df.columns if col not in cat_cols + ['target']]
    emb_sizes = [(df[col].astype('category').cat.codes.max() + 1, min(50, (df[col].nunique() + 1) // 2)) for col in cat_cols]
    return df, test, cat_cols, cont_cols, emb_sizes

def process_data(data, cat_cols, cont_cols):
    cats = np.stack([data[col].astype('category').cat.codes.values for col in cat_cols], axis=1)
    conts = np.stack([data[col].values for col in cont_cols], axis=1)
    scaler = StandardScaler()
    conts = scaler.fit_transform(conts)
    y = data['target'].map({'Donor': 1, 'No Donor': 0}).values
    return torch.tensor(cats, dtype=torch.int64), torch.tensor(conts, dtype=torch.float), torch.tensor(y, dtype=torch.long)

class TabularModel(nn.Module):
    def __init__(self, emb_sizes, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_sizes])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        total_emb_size = sum(nf for _, nf in emb_sizes)
        n_in = total_emb_size + n_cont
        layerlist = nn.ModuleList()
        for output_size in layers:
            layerlist.append(nn.Linear(n_in, output_size))
            layerlist.append(nn.ReLU())
            layerlist.append(nn.BatchNorm1d(output_size))
            layerlist.append(nn.Dropout(p))
            n_in = output_size
        layerlist.append(nn.Linear(layers[-1], out_sz))
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        embeddings = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        return self.layers(x)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose and self.counter == self.patience:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}, stopping...')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def main():
    args = setup_args()
    os.chdir('/home/dan/FUNDRAISING')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df, test, cat_cols, cont_cols, emb_sizes = load_and_preprocess_data('./data/df.csv')
    cats, conts, targets = process_data(df, cat_cols, cont_cols)
    dataset = TensorDataset(cats, conts, targets)

    model = TabularModel(emb_sizes, len(cont_cols), 2, args.layers, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_subsampler)

        for epoch in range(args.epochs):
            model.train()
            for cats, conts, y in train_loader:
                cats, conts, y = cats.to(device), conts.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(cats, conts)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_accs = []
                for cats, conts, y in val_loader:
                    cats, conts, y = cats.to(device), conts.to(device), y.to(device)
                    outputs = model(cats, conts)
                    val_accs.append((outputs.argmax(dim=1) == y).float().mean().item())
                avg_val_acc = np.mean(val_accs)
            early_stopping(avg_val_acc, model)
            if early_stopping.early_stop:
                break
        results.append(avg_val_acc)

    average_val_acc = np.mean(results)
    print(json.dumps({
        'average_val_accuracy': average_val_acc,
        'epochs': args.epochs,
        'patience': args.patience,
        'learning_rate': args.lr,
        'layers': args.layers,
        'dropout': args.dropout,
        'folds': args.folds}))


    preds = []
    model.eval()
    test_cats, test_conts, test_targets = process_data(test)
    test_dataset = TensorDataset(test_cats, test_conts, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    with torch.no_grad():
        for cats, conts, y in test_loader:
            cats, conts, y = cats.to(device), conts.to(device), y.to(device)
            output = model(cats, conts)
            predicted = output.argmax(dim=1)  # Ensure you use dim=1
            # print(predicted)
            preds.append(predicted.cpu())  # Move predictions to CPU
    # print(type(preds[0]))
    # Concatenate all batch predictions into a single tensor
    # preds = torch.cat(preds)
    preds = preds[0].tolist()
    # print(preds)
    final_preds = []
    donor=1
    no_donor=1
    for i in preds:
        if i == 1:
            final_preds.append('Donor')
            donor +=1
        else:
            final_preds.append('No Donor')
            no_donor+=1
    print(f'% Donor = {donor / (donor+no_donor)}')

    save_df = pd.DataFrame(final_preds, columns=['values'])
    save_df.to_csv(f'./preds/bashing/{args.i}_preds.csv', index=False)

    



if __name__ == "__main__":
    main()
