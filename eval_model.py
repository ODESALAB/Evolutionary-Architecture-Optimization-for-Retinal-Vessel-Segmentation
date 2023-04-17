import glob
import pickle
import os
import io
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils.losses import *
from utils.metrics import *
from torch.utils.data import DataLoader
from utils.dataset import vessel_dataset
from models.unet import *
from utils.save_best_model import BestModelCheckPoint

def getBestModelNumbers(path="results"):
    result = []
    for file in glob.glob(f"{path}/*.pkl"):
        with open(file, "rb") as f:
            data = GPU_Unpickler(f).load()
            result.append((data.fitness, data.cost, data.solNo))

    return sorted(result, key=lambda x: x[0])[-5:]

class GPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cuda')
        else:
            return super().find_class(module, name)

def readPickleFile(file, path="results"):
    with open(f"{path}/model_{file}.pkl", "rb") as f:
        data = pickle.load(f)
    
    return data


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


import warnings
warnings.filterwarnings("ignore")

path = "results/ode_drive_patch_42"
data_path = "DataSets/DRIVE"
batch_size = 128


seed = 0
modelNo = 517

seed_torch(seed)

checkpoint = BestModelCheckPoint(modelNo)
device = torch.device('cuda')

model = None
with open(f"{path}/model_{modelNo}.pkl", "rb") as f:
    model = GPU_Unpickler(f).load()

print("Model No:", model.solNo, "Seed:", seed)

model.reset()
model.to(device)

train_dataset = vessel_dataset(data_path, mode="training", split=0.9)
val_dataset = vessel_dataset(data_path, mode="training", split=0.9, is_val=True)
test_dataset = vessel_dataset(data_path, mode="test")
val_loader = DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True, drop_last=False)
train_loader = DataLoader(train_dataset, batch_size, shuffle=False, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, 1, shuffle=False, pin_memory=True)

print(train_dataset.__len__())

optimizer = optim.Adam(model.parameters(), lr=1e-3) # 1e-3

#loss = DiceLoss()
loss = CombinedLoss(loss_type='jaccard')
metric = DiceCoef()
iou_metric = IoU()

log = ""

for epoch in range(200): 
  train_loss = []
  train_dice = []
  
  # Train Phase
  model.train()
  for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    
    with torch.set_grad_enabled(True):
      output = model(inputs)
      error = loss(output, labels)
      train_loss.append(error.item())
      train_dice.append(metric(output, labels).item())
      optimizer.zero_grad()
      error.backward()
      optimizer.step()
      del output
      
      del error
    del inputs
    del labels

  torch.cuda.empty_cache()

  # Validation Phase
  val_loss = []
  val_dice = []
  model.eval()
  for inputs, labels in val_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    output = model(inputs)
    error = loss(output, labels)
    val_loss.append(error.item())
    val_dice.append(metric(output, labels).item())

    del output
    del error
    del inputs
    del labels
    
  avg_tr_loss = sum(train_loss) / len(train_loss)
  avg_tr_score = sum(train_dice) / len(train_dice)
  avg_val_loss = sum(val_loss) / len(val_loss)
  avg_val_score = sum(val_dice) / len(val_dice)
  txt = f"\nEpoch: {epoch}, tr_loss: {avg_tr_loss}, tr_dice_score: {avg_tr_score}, val_loss: {avg_val_loss}, val_dice: {avg_val_score}"
  log += txt
  print(txt)
  checkpoint.check(avg_val_score, model, seed)
  torch.cuda.empty_cache()

# Get Best Model
print("Load Model...")
model.load_state_dict(torch.load(f"model_{modelNo}_seed_{seed}.pt"))
model.to(device)

# Testing
test_loss = []
test_dice = []
test_iou = []

model.eval()

for inputs, labels in test_loader:
    with torch.no_grad():
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        error = loss(output, labels)
        test_dice.append(metric(output, labels).item())
        test_iou.append(iou_metric(output, labels).item())
        test_loss.append(error.item())

log += f"\ntest_loss: {sum(test_loss) / len(test_loss)}, test_dice: {sum(test_dice) / len(test_dice)}, test_iou: {sum(test_iou) / len(test_iou)}"
print(f"test_loss: {sum(test_loss) / len(test_loss)}, test_dice: {sum(test_dice) / len(test_dice)}, test_iou: {sum(test_iou) / len(test_iou)}")

# Write Log
with open(f"log_{modelNo}_seed_{seed}.txt", "w") as f:
    f.write(log)

torch.cuda.empty_cache()
