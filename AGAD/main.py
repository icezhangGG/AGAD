# ---------  YI ------------
# Time 2023
import math
import sys,os
from numpy.matrixlib.defmatrix import matrix
sys.path.append(os.getcwd())
from Process.process import * 
#from Process.process_user import *
import torch as th
import torch.nn as nn
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from others.earlystopping import EarlyStopping
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from others.evaluate import *
from Adv.adv import *
from Att.att import *

from Gnn.GAT import *

from torch_geometric.nn import GATConv
import copy
import random


def setup_seed(seed):
     th.manual_seed(seed)
     th.cuda.manual_seed_all(seed) 
     np.random.seed(seed)
     random.seed(seed)
     th.backends.cudnn.deterministic = True

setup_seed(3040)


class hard_fc(th.nn.Module):
    def __init__(self, d_in,d_hid, DroPout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(DroPout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

def train_Model(x_test, x_train, lr, weight_decay, patience, n_epochs,batchsize,dataname):

    model = GAT_Net(768,64,64,batchsize).to(device)

    adv = ADC(model)
    for para in model.hard_fc1.parameters():
        para.requires_grad = False
    for para in model.hard_fc2.parameters():
        para.requires_grad = False
    optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    

    for para in model.hard_fc1.parameters():
        para.requires_grad = True
    for para in model.hard_fc2.parameters():
        para.requires_grad = True

    optimizer_hard = th.optim.SGD([{'params': model.hard_fc1.parameters()},
                                    {'params': model.hard_fc2.parameters()}], lr=0.001)

    model.train() 
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True) 

    for epoch in range(n_epochs): 
        traindata_list, testdata_list = loadData(dataname, x_train, x_test, droprate=0.4) # T15 droprate = 0.1
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)     
        avg_loss = [] 
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        NUM=1
        beta=0.3
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            out_labels, cl_loss, y = model(Batch_data) 
            finalloss = F.nll_loss(out_labels,y)
            loss = finalloss + beta*cl_loss
            avg_loss.append(loss.item())


            optimizer.zero_grad()
            loss.backward()

            adv.attack()
            out_labels, cl_loss, y = model(Batch_data) 
            finalloss = F.nll_loss(out_labels,y)
            loss_adv = finalloss + beta*cl_loss
            loss_adv.backward()

            adv.restore()
            optimizer.step()
            ##--------------------------------##


            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(y).sum().item()
            train_acc = correct / len(y) 
            avg_acc.append(train_acc)
            print("Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch, batch_idx,loss.item(),train_acc))
            batch_idx = batch_idx + 1
            NUM += 1
            #print('train_loss: ', loss.item())
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        
        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval() 
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            val_out, val_cl_loss, y = model(Batch_data)
            val_loss = F.nll_loss(val_out, y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(y).sum().item()
            val_acc = correct / len(y) 
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, y) 
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f} | Val_Accuracy {:.4f}".format(epoch, np.mean(avg_loss), np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))
        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)

        if epoch > 25:
            early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                        np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'My modle', dataname)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        print("NR F1: {:.3f}".format(F1),"FR F2: {:.3f}".format(F2),"TR F3: {:.3f}".format(F3),"UR F4: {:.3f}".format(F4))
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return accs,F1,F2,F3,F4


##---------------------------------main---------------------------------------
if __name__ == '__main__':
    scale = 1
    lr=0.0005 * scale
    weight_decay=1e-4
    patience=10
    n_epochs=200
    batchsize=120
    datasetname='Twitter16' # (1)Twitter15  (2)pheme  (3)weibo
    #model="GCN"
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    test_accs = []
    NR_F1 = [] # NR
    FR_F1 = [] # FR
    TR_F1 = [] # TR
    UR_F1 = [] # UR

    data_path = './data/twitter16/'
    laebl_path = './data/Twitter16_label_All.txt'

    fold0_x_test, fold0_x_train, \
    fold1_x_test,  fold1_x_train,\
    fold2_x_test, fold2_x_train, \
    fold3_x_test, fold3_x_train, \
    fold4_x_test,fold4_x_train = load5foldData(datasetname,data_path,laebl_path)

    print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train))
    print('fold1 shape: ', len(fold1_x_test), len(fold1_x_train))
    print('fold2 shape: ', len(fold2_x_test), len(fold2_x_train))
    print('fold3 shape: ', len(fold3_x_test), len(fold3_x_train))
    print('fold4 shape: ', len(fold4_x_test), len(fold4_x_train))


    accs0, F1_0, F2_0, F3_0, F4_0 = train_Model(fold0_x_test,fold0_x_train,lr,weight_decay, patience,n_epochs,batchsize,datasetname)
    accs1, F1_1, F2_1, F3_1, F4_1 = train_Model(fold1_x_test,fold1_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
    accs2, F1_2, F2_2, F3_2, F4_2 = train_Model(fold2_x_test,fold2_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
    accs3, F1_3, F2_3, F3_3, F4_3 = train_Model(fold3_x_test,fold3_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
    accs4, F1_4, F2_4, F3_4, F4_4 = train_Model(fold4_x_test,fold4_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
    test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
    NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
    FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
    UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
    print("AVG_result: {:.4f}|UR F1: {:.4f}|NR F1: {:.4f}|TR F1: {:.4f}|FR F1: {:.4f}".format(sum(test_accs), sum(NR_F1), sum(FR_F1), sum(TR_F1), sum(UR_F1)))
