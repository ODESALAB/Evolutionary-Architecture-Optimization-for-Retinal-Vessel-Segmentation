import glob
import pickle
import os
import torch
import random
import torch.nn as nn
import numpy as np
import torch.optim as optim
from scores import *
from models.unet import *
from utils.losses import *
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils.dataset import vessel_dataset

from utils.save_best_model import BestModelCheckPoint

def getBestModelNumbers(path):
    result = []
    for file in glob.glob(f"results/{path}/*.pkl"):
        with open(file, "rb") as f:
            data = pickle.load(f)
            result.append((data.fitness, data.cost, data.solNo))

    return sorted(result, key=lambda x: x[0])[-5:]

def readPickleFile(file, path):
    with open(f"results/{path}/model_{file}.pkl", "rb") as f:
        data = pickle.load(f)
    
    return data

seed = 0
modelNo = 563
path = "ga_chasedb1_patch_42"
data_path = "DataSets/CHASEDB1"

for seed in [0, 42, 143, 1234, 3074]:

    print(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    checkpoint = BestModelCheckPoint(modelNo)
    device = torch.device('cuda:1')

    model = readPickleFile(modelNo, path)
    #model = build_unet()
    #print("Model No:", model.solNo, "Seed:", seed)

    test_dataset = vessel_dataset(data_path, mode="test")
    test_loader = DataLoader(test_dataset, 1, shuffle=False,  num_workers=0, pin_memory=True)

    import matplotlib.pyplot as plt

    if not os.path.exists(f"figures/{path}/model_{modelNo}_seed_{seed}/"):
        os.makedirs(f"figures/{path}/model_{modelNo}_seed_{seed}/")

    print("Load Model...")
    model.load_state_dict(torch.load(f"results/{path}/long_runs/model_{modelNo}_seed_{seed}.pt"))
    model.to(device)

    metrics = {"JAC": iou_score, "IOU": iou_frnet, "DICE": f1_score, "SNS" : sensitivity, "SP": specificity,
                "SB": sensibility, "GCE": global_consistency_error, "CNF": conformity, "F1": f1_score_frnet,
                "ACC": accuracy, "PRC": positive_predictive_value, "REC": recall, "AUC_2": auc_score,
                "RI": rand_index, "ARI": adjusted_rand_index, "MI": mutual_information,
                "VOI": variation_of_information, "ICC": interclass_correlation, "PBD": probabilistic_distance,
                "KAP": cohens_cappa, "HD95": hausdorff_distance_95th_quantile, "AUC": roc_auc, "AHD": average_hausdorff_distance,
                "MHD": mahalanobis_distance
                }

    model.eval()
    counter = 1
    df_method = pd.DataFrame(index=range(1, 21), columns=metrics.keys())
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        output = torch.sigmoid(output) 
        
        tp, fp, fn, tn = smp.metrics.get_stats(output, labels.round().long(), mode='binary', threshold=0.5)
        for lbl, method in metrics.items():
            reduction = 'micro'
            if lbl in ['RI', 'ICC', 'PBD', 'HD95', 'AUC', 'AHD', 'MHD', "AUC_2"]:
                continue
                #score = method(labels.cpu().data.numpy().reshape((1008, 1008)), output.cpu().data.numpy().reshape((1008, 1008)))
            else:
                if lbl in ['ACC']: reduction = 'macro'
                elif lbl in ['REC', 'SNS']: reduction = 'micro-imagewise'
                score = method(tp, fp, fn, tn).tolist()[0][0]
            
            df_method.loc[counter, lbl] = score

        
        
        img = output.cpu().data.numpy().reshape((1008, 1008))
        
        plt.imsave(f"figures/{path}/model_{modelNo}_seed_{seed}/{counter}.png", img, cmap='gray')
        counter += 1
        del output

    df_method.to_excel(f"figures/{path}/model_{modelNo}_seed_{seed}.xlsx")
    plt.close()
    