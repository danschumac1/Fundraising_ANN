"""
Created on 05/05/2024

@author: Dan Schumacher

THIS FILE IS FOR DR.CAMPBELL'S PREDICTIVE MODELING CLASS
THE AIM IS TO CREATE A NN THAT CAN CLASSIFY A LIST OF PEOPLE INTO DONORS/NON-DONORS.
"""

# =============================================================================
# SET-UP AND IMPORTS
# =============================================================================
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, TensorDataset

# Set directory
os.chdir('/home/dan/FUNDRAISING')

# SET UP CUDA
# Check if GPU is available and set the default device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# PREPARE DATA
# =============================================================================

# GLOBAL CHANGES
# Load data
df = pd.read_csv('./data/df.csv', index_col='Unnamed: 0')

zipc = []
for i, row in df.iterrows():
    if row['zipconvert2'] == 'Yes':
        zipc.append('zc2')
    elif row['zipconvert3'] == 'Yes':
        zipc.append('zc3')
    elif row['zipconvert4'] == 'Yes':
        zipc.append('zc4')
    elif row['zipconvert5'] == 'Yes':
        zipc.append('zc5')
    else:
        zipc.append(None)

# set it to dataframe
df['zipconvert'] = zipc
df['zipconvert'] = df['zipconvert'].fillna('unknown')

# drop other rows
df = df.drop(['zipconvert2', 'zipconvert3', 'zipconvert4', 'zipconvert5'], axis =1)

# Split data first
train = df[df['type'] == 'train']
dev = df[df['type'] == 'dev']
test = df[df['type'] == 'test']

# Remove the 'type' column after splitting to avoid leakage
train = train.drop('type', axis=1,)
dev = dev.drop('type', axis=1,)
test = test.drop('type', axis=1,)

# Categorical and continuous columns
cat_cols = ['homeowner', 'female', 'zipconvert', 'wealth', 'income', 'num_child']
cont_cols = [col for col in train.columns if col not in cat_cols]
cont_cols.remove('target')

# Function to process datasets
def process_data(data):
    # Convert categorical columns
    cats = np.stack([data[col].astype('category').cat.codes.values for col in cat_cols], axis=1)
    cats = torch.tensor(cats, dtype=torch.int64)

    # Convert continuous columns
    conts = np.stack([data[col].values for col in cont_cols], axis=1)
    conts = torch.tensor(conts, dtype=torch.float)

    # Target column
    y = torch.tensor(data['target'].map({'Donor': 1, 'No Donor': 0}).values).long()

    return cats, conts, y

# Process data and then move each tensor to the GPU individually
cat_train, con_train, y_train = process_data(train)
cat_train, con_train, y_train = cat_train.to(device), con_train.to(device), y_train.to(device)

cat_dev, con_dev, y_dev = process_data(dev)
cat_dev, con_dev, y_dev = cat_dev.to(device), con_dev.to(device), y_dev.to(device)

cat_test, con_test, y_test = process_data(test)
cat_test, con_test, y_test = cat_test.to(device), con_test.to(device), y_test.to(device)


# =============================================================================
# DEFINE A TABULAR MODEL
# =============================================================================
class TabularModel(nn.Module):

    def __init__(self, emb_sizes, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_sizes])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = []
        n_emb = sum((nf for ni,nf in emb_sizes))
        n_in = n_emb + n_cont
        
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
            
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        embeddings = []
        for i,e in enumerate(self.embeds):
            # print(f"Max index for embedding {i}: {x_cat[:,i].max()}")  # Debug: print max index

            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x

# Calculate embedding sizes
emb_sizes = [(df[col].astype('category').cat.codes.max()+1, min(50, (df[col].astype('category').cat.codes.max()+2)//2)) for col in cat_cols]

# print("\n\nCategory columns and their embedding sizes:\n\n")
# for col, size in zip(cat_cols, emb_sizes):
#     print(f"{col}: {size}")


# for col in cat_cols:
#     print(f"{col} unique values: {df[col].unique()}")


model = TabularModel(emb_sizes, len(cont_cols), 2, [200,200,200, 100], p=0.4)
model = model.to(device)
# =============================================================================
# LOSS FUNCTION AND OPTIMIZER
# =============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# # =============================================================================
# # TRAIN MODEL
# # =============================================================================
train_dataset = TensorDataset(cat_train, con_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

dev_dataset = TensorDataset(cat_dev, con_dev, y_dev)
dev_loader = DataLoader(dev_dataset, batch_size=2048, shuffle=False)

epochs = 10
train_losses = []
validation_losses = []
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    train_loss = 0

    for cats, conts, y in train_loader:
        cats, conts, y = cats.to(device), conts.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(cats, conts)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Average the loss over all batches
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation step
    model.eval()  # Set the model to evaluation mode
    validation_loss = 0
    with torch.no_grad():
        for cats, conts, y in dev_loader:
            cats, conts, y = cats.to(device), conts.to(device), y.to(device)
            output = model(cats, conts)
            loss = criterion(output, y)
            validation_loss += loss.item()
    
    # Average the loss over all validation batches
    validation_loss /= len(dev_loader)
    validation_losses.append(validation_loss)

    if epoch % 10 == 0:  # Print every 100 epochs
        print(f'Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}')

# =============================================================================
# EVALUATE MODEL
# =============================================================================

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss.png')  # Saves the plot as a PNG file
plt.close()  # Close the plot to free up memory



# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for cats, conts, y in dev_loader:
#         cats, conts, y = cats.to(device), conts.to(device), y.to(device)
#         output = model(cats, conts)
#         predicted = output.argmax(1)
#         total += y.size(0)
#         correct += (predicted == y).sum().item()
#     print(f'Accuracy: {100 * correct / total}%')


# import math

# # Initial setup
# low_lr = 1e-6
# high_lr = 1e-1
# total_batches = 100  # Number of batches to test over
# lr_lambda = lambda x: math.exp(x * math.log(high_lr / low_lr) / total_batches)
# lr_range = [low_lr * lr_lambda(i) for i in range(total_batches)]
# train_losses = []
# lrs = []

# model.train()
# for batch_idx, (cats, conts, y) in enumerate(train_loader):
#     if batch_idx >= total_batches:
#         break
#     lr = lr_range[batch_idx]
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
    
#     cats, conts, y = cats.to(device), conts.to(device), y.to(device)
#     optimizer.zero_grad()
#     output = model(cats, conts)
#     loss = criterion(output, y)
#     loss.backward()
#     optimizer.step()

#     train_losses.append(loss.item())
#     lrs.append(lr)

#     print(f'Batch: {batch_idx + 1}, LR: {lr:.6f}, Loss: {loss.item():.4f}')

# # Plotting the result
# plt.figure(figsize=(10, 5))
# plt.plot(lrs, train_losses)
# plt.xscale('log')
# plt.xlabel('Learning Rate (log scale)')
# plt.ylabel('Loss')
# plt.title('Learning Rate Finder')
# plt.savefig('lr_finder.png')
# plt.close()
