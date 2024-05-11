import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

import os
import torch
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
#endregion
#region # CUDA AND OS SETUP
# =============================================================================
# CUDA AND OS SETUP
# =============================================================================

# Set directory and device setup
os.chdir('/home/dan/FUNDRAISING')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#endregion
#region # HELPER FUNCTIONS
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_vif(X):
    """
    Calculates Variance Inflation Factors (VIF) for each feature in a dataset.
    This function adds a constant term to the predictor matrix and computes VIF for each feature.
    """
    X_with_const = sm.add_constant(X)
    vif_vals = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    return pd.DataFrame({'VIF': vif_vals, 'col': X_with_const.columns})

#endregion
#region # DATACLEANING
# =============================================================================
# DATACLEANING
# =============================================================================
# Load and preprocess data
df = pd.read_csv('./data/df.csv').drop('Unnamed: 0', axis=1)
test = df.query("type == 'test'").copy()
df = df.query("type != 'test'").copy()

# Convert categorical columns to binary
columns_to_binary = ['zipconvert2', 'zipconvert3', 'zipconvert4', 'zipconvert5', 'homeowner', 'female']
for col in columns_to_binary:
    df[col] = (df[col] == 'Yes').astype(int)
    test[col] = (test[col] == 'Yes').astype(int)

df['target'] = (df['target'] == 'Donor').astype(int)
test['target'] = (test['target'] == 'Donor').astype(int)

# Drop unused columns
df.drop(['type'], axis=1, inplace=True)
test.drop(['type'], axis=1, inplace=True)

# Add transformations
def add_transformations(data, cont_cols):
    for var in cont_cols:
        data[f'log_{var}'] = np.log(data[var] + 1)
        data[f'sq_{var}'] = data[var]**2
        data[f'sqrt_{var}'] = np.sqrt(data[var])
        data[f'inv_{var}'] = 1 / (data[var] + 1)
        data[f'boxcox_{var}'], _ = stats.boxcox(data[var] + 1)
        data[f'sigmoid_{var}'] = 1 / (1 + np.exp(-data[var]))
        data[f'sin_{var}'] = np.sin(data[var])
        data[f'cos_{var}'] = np.cos(data[var])

cont_cols = [col for col in df.columns if col not in columns_to_binary + ['target']]
add_transformations(df, cont_cols)
add_transformations(test, cont_cols)

# Feature selection based on correlation
correlated_features = df.corr().abs().nlargest(20, 'target').index
df_subset = df[correlated_features]
test_subset = test[correlated_features]

df_subset.columns
'num_prom','boxcox_num_prom', 'sqrt_num_prom' ,'log_num_prom'
# VIFF
X = df_subset.drop(['target','sqrt_months_since_donate','months_since_donate','log_last_gift','log_avg_gift','sq_months_since_donate','sqrt_avg_gift','log_months_since_donate','boxcox_largest_gift','sqrt_last_gift','boxcox_avg_gift','boxcox_last_gift','inv_largest_gift'], axis = 1)
y = df_subset['target']
vif = calculate_vif(X)
# vif['VIF'].max()

vif[vif['VIF'] == vif['VIF'].drop(0).max()]
vif
kept2 = list(vif['col'].values)[1:]

# Final dataset preparation
df = df_subset[kept2 + ['target']].copy()
test = test_subset[kept2 + ['target']].copy()
for col in df.columns:
    print(col)

#endregion
#region # CLEANING FOR NN
# =============================================================================
# CLEANING FOR NN
# =============================================================================

cont_cols = [col for col in df.columns if col not in columns_to_binary + ['target']]


# Prepare neural network data
def process_data(data, cont_cols):
    conts = np.stack([data[col].values for col in cont_cols], axis=1)
    scaler = RobustScaler()
    conts = scaler.fit_transform(conts)
    targets = data['target'].values
    return torch.tensor(conts, dtype=torch.float), torch.tensor(targets, dtype=torch.long)


conts, targets = process_data(df_subset, cont_cols)
dataset = TensorDataset(conts, targets)

test_conts, test_targets = process_data(test, cont_cols)
test_dataset = TensorDataset(test_conts, test_targets)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

# =============================================================================
# DEFINE A TABULAR MODEL
# =============================================================================
class TabularModel(nn.Module):
    def __init__(self, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = nn.ModuleList()
        n_in = n_cont
        for output_size in layers:
            layerlist.append(nn.Linear(n_in, output_size))
            layerlist.append(nn.ReLU())
            layerlist.append(nn.BatchNorm1d(output_size))
            layerlist.append(nn.Dropout(p))
            n_in = output_size
        
        layerlist.append(nn.Linear(layers[-1], out_sz))
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cont):
        x_cont = self.bn_cont(x_cont)
        return self.layers(x_cont)

# =============================================================================
# CROSS-VALIDATION SETUP
# =============================================================================
def calculate_accuracy(y_pred, y_true):
    y_pred_classes = torch.argmax(y_pred, dim=1)
    correct = (y_pred_classes == y_true).float()
    acc = correct.sum() / len(correct)
    return acc

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=128, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=128, sampler=val_subsampler)

    model = TabularModel(len(cont_cols), 2, [150, 150, 150], p=.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=.01)

    for epoch in range(100):
        model.train()
        total_loss, total_acc = 0, 0
        for conts, y in train_loader:
            conts, y = conts.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(conts)
            loss = criterion(outputs, y)
            acc = calculate_accuracy(outputs, y)
            total_loss += loss.item()
            total_acc += acc.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader)

        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for conts, y in val_loader:
                conts, y = conts.to(device), y.to(device)
                outputs = model(conts)
                loss = criterion(outputs, y)
                acc = calculate_accuracy(outputs, y)
                val_loss += loss.item()
                val_acc += acc.item()
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        print(f'Fold {fold+1}, Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')
        
        results.append((avg_val_loss, avg_val_acc))

average_val_loss = sum(x[0] for x in results) / len(results)
average_val_acc = sum(x[1] for x in results) / len(results)
print(f'Average Validation Loss: {average_val_loss:.4f}, Average Validation Accuracy: {average_val_acc:.4f}')

# =============================================================================
# POST-PROCESSING
# =============================================================================
# Here you might want to process your predictions after evaluation
preds = []
model.eval()

with torch.no_grad():
    for conts, y in test_loader:
        conts, y = conts.to(device), y.to(device)
        output = model(conts)
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
save_df.to_csv('./preds/preds.csv', index=False)