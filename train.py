import seaborn as sns
from torch_geometric.data import DataLoader
from fMRIdataset import fMRI
import tqdm
import pdb
import torch
import numpy as np
from model import *
from itertools import chain
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dataset = fMRI(root="data/", filename="248_292_train.csv")
data = fMRI(root="data/", filename="96_292_test.csv", test=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(data, batch_size=32, shuffle=True)
model = GraphNetwork(64).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# loss_fn = torch.nn.NLLLoss()
loss_fn_1 = torch.nn.MSELoss()
loss_fn_2 = torch.nn.L1Loss()
loss_fn = torch.nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=15)

def train(loader):
    # Enumerate over the data
    
    batch_loss = 0
    for idx, batch in enumerate(loader):
      # Use GPU
      batch.to(device)  
      # Reset gradients
      optimizer.zero_grad() 
      # Passing the node features and the connection info
      pred = model(batch) 
      # Calculating the loss and gradients
      loss = 0.5 * loss_fn_1(torch.squeeze(pred), torch.squeeze(batch.y).float()) 
      loss += 0.5 * loss_fn_2(torch.squeeze(pred), torch.squeeze(batch.y).float()) 
    #   loss = loss_fn(torch.squeeze(pred), torch.squeeze(batch.y).float()) 
      loss.backward()  
      # Update using the gradients
      optimizer.step()  
      batch_loss += loss.item() 
    return batch_loss/(idx+1)

def test(loader):
    # Enumerate over the data
    batch_loss = 0
    for idx, batch in enumerate(loader):
      # Use GPU
      batch.to(device)  
      # Reset gradients
      # Passing the node features and the connection info
      model.eval()
      pred = model(batch) 
      # Calculating the loss and gradients
      loss = 0.5 * loss_fn_1(torch.squeeze(pred), torch.squeeze(batch.y).float()) 
      loss += 0.5 * loss_fn_2(torch.squeeze(pred), torch.squeeze(batch.y).float()) 
    #   loss = loss_fn(torch.squeeze(pred), torch.squeeze(batch.y).float()) 
      batch_loss += loss.item() 
    return batch_loss/(idx+1)

def inference(loader):
    predictions = []
    gts = []
    sexs = []
    for idx, batch in enumerate(loader):
        # Use GPU
        batch.to(device)  
        model.eval()
        # Reset gradients
        # Passing the node features and the connection info
        pred = model(batch) 
        predictions.append(np.int16(np.array(torch.squeeze((pred.detach()))).tolist()))
        gts.append(np.array(batch.y).tolist())
        sexs.append(np.array(batch.sex).tolist())
    # predictions = list(predictions)
    predictions = list(chain(*predictions))
    gts = list(chain(*gts))
    sexs = list(chain(*sexs))
    
    
    return predictions, gts, sexs

print("Starting training...")
losses = []
for epoch in range(1000):
    batch_ave_train_loss = train(train_loader)
    batch_ave_test_loss = test(test_loader)
    if epoch % 2 == 0:
        
        predictions, gt, sexs = inference(test_loader)
        predictions = np.array(predictions)
        gt = np.array(gt)
        sns.scatterplot(x=gt, y=predictions, hue=sexs)
        plt.xlabel('AGE')
        plt.ylabel('Prediction')
        plt.xlim([gt.min()-5, gt.max()+5])
        plt.ylim([gt.min()-5, gt.max()+5])
        plt.text(5, 50, f'Epoch: {epoch}', color='red', fontsize=12, ha='center', va='center', weight='bold')
        plt.savefig(f'./figure/test/{epoch}_test.png')
        plt.close()
        
        predictions, gt, sexs = inference(train_loader)
        predictions = np.array(predictions)
        gt = np.array(gt)
        sns.scatterplot(x=gt, y=predictions, hue=sexs)
        plt.xlabel('AGE')
        plt.ylabel('Prediction')
        plt.xlim([gt.min()-5, gt.max()+5])
        plt.ylim([gt.min()-5, gt.max()+5])
        plt.text(5, 65, f'Epoch: {epoch}', color='red', fontsize=12, ha='center', va='center', weight='bold')
        plt.savefig(f'./figure/train/{epoch}_train.png')
        plt.close()
        
        print(f"Epoch {epoch} | Train Loss {batch_ave_train_loss :.4f} | Test Loss {batch_ave_test_loss :.4f} | Current Learning Rate {optimizer.param_groups[-1]['lr']}")
    scheduler.step(batch_ave_test_loss)
    
# torch.save(model.state_dict(), './trained_model/weight.pth')
# torch.save(model, './trained_model/model.pth')
