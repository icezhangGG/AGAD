import sys,os
from numpy.matrixlib.defmatrix import matrix
sys.path.append(os.getcwd())
from Process.process_pheme import *
#from Process.process_user import *
import torch as th
import torch.nn as nn
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from others.earlystopping import EarlyStopping2
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold_pheme import *
from others.evaluate import *
from torch_geometric.nn import GATConv
import random
#from Gnn.GAT import *
from Gnn.SGAT import *
from Att.att import *
from Adv.adv import *



class hard_fc(th.nn.Module):
    def __init__(self, d_in,d_hid, DroPout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid, bias=False) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in, bias=False) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(DroPout)
       
    def forward(self, x):

        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

    
        return x





def train_GAT(x_test, x_train,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter): 
    model = SGAT_Net(768,64,64).to(device)

    adv = ADV(model)

    optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    


    train_losses = [] 
    val_losses = [] 
    train_accs = [] 
    val_accs = [] 
    early_stopping = EarlyStopping2(patience=patience, verbose=True)


    for epoch in range(n_epochs): 
        traindata_list, testdata_list = loadUdData(dataname, x_train, x_test, droprate=0.3) 
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=10)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=10)       
        avg_loss = [] 
        avg_acc = []
        batch_idx = 0
        model.train()
        #tqdm_train_loader = tqdm(train_loader)
        NUM=1
        for Batch_data in train_loader:
            Batch_data.to(device)
            out_labels, cl_loss, y = model(Batch_data)
            finalloss=F.nll_loss(out_labels,y)
            loss = finalloss + 0.3*cl_loss
            avg_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
########################adv###########################

            adv.attack()
            out_labels, cl_loss, y = model(Batch_data)

            finalloss = F.nll_loss(out_labels, y)
            loss_adv = finalloss + 0.3 * cl_loss
            loss_adv.backward()

            adv.restore()
########################adv###########################

            optimizer.step()

            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(y).sum().item()
            train_acc = correct / len(y) 
            avg_acc.append(train_acc)
            #print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
            #                                                                                     loss.item(),
            #                                                                                     train_acc))
            batch_idx = batch_idx + 1
            NUM += 1

        train_losses.append(np.mean(avg_loss)) 
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval() 
        #tqdm_test_loader = tqdm(test_loader)
        for Batch_data in test_loader:
            Batch_data.to(device)
            val_out, val_cl_loss, y = model(Batch_data)
            val_loss = F.nll_loss(val_out, y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1) 
            correct = val_pred.eq(y).sum().item()
            val_acc = correct / len(y) 
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluation2class(
                val_pred, y)

            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_Acc_all), np.mean(temp_val_Acc1),
                       np.mean(temp_val_Acc2), np.mean(temp_val_Prec1),
                       np.mean(temp_val_Prec2), np.mean(temp_val_Recll1), np.mean(temp_val_Recll2),
                       np.mean(temp_val_F1),
                       np.mean(temp_val_F2), model, 'AGAD')
        accs = np.mean(temp_val_Acc_all)
        acc1 = np.mean(temp_val_Acc1)
        acc2 = np.mean(temp_val_Acc2)
        pre1 = np.mean(temp_val_Prec1)
        pre2 = np.mean(temp_val_Prec2)
        rec1 = np.mean(temp_val_Recll1)
        rec2 = np.mean(temp_val_Recll2)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.temp_val_Acc_all
            acc1 = early_stopping.temp_val_Acc1
            acc2 = early_stopping.temp_val_Acc2
            pre1 = early_stopping.temp_val_Prec1
            pre2 = early_stopping.temp_val_Prec2
            rec1 = early_stopping.temp_val_Recll1
            rec2 = early_stopping.temp_val_Recll2
            F1 = early_stopping.temp_val_F1
            F2 = early_stopping.temp_val_F2
            break
    return train_losses, val_losses, train_accs, val_accs, accs, acc1, pre1, rec1, F1, acc2, pre2, rec2, F2



##---------------------------------main---------------------------------------

if __name__ == '__main__':

    def setup_seed(seed):
         th.manual_seed(seed)
         th.cuda.manual_seed_all(seed)
         np.random.seed(seed)
         random.seed(seed)
         th.backends.cudnn.deterministic = True

    setup_seed(3040)
    print('seed=3040')


    lr=0.0005
    weight_decay=1e-4
    patience=15
    n_epochs=200
    batchsize=200 # twitter

    datasetname='Pheme'
    iterations=1
    model="GAT"
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    #device = th.device('cpu')


    fold0_x_test, fold0_x_train,\
        fold1_x_test, fold1_x_train, \
        fold2_x_test, fold2_x_train,  \
        fold3_x_test, fold3_x_train,  \
        fold4_x_test, fold4_x_train = load5foldData(datasetname)

    print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train))
    print('fold1 shape: ', len(fold1_x_test), len(fold1_x_train))
    print('fold2 shape: ', len(fold2_x_test), len(fold2_x_train))
    print('fold3 shape: ', len(fold3_x_test), len(fold3_x_train))
    print('fold4 shape: ', len(fold4_x_test), len(fold4_x_train))


    test_accs,ACC1,ACC2,PRE1,PRE2,REC1,REC2,F1,F2 = [],[],[],[],[],[],[],[],[]

    for iter in range(iterations):
        train_losses, val_losses, train_accs, val_accs, accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = train_GAT(
                                                                                                   fold0_x_test,
                                                                                                   fold0_x_train,

                                                                                                   lr, weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
        train_losses, val_losses, train_accs, val_accs,accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = train_GAT(
                                                                                                   fold1_x_test,
                                                                                                   fold1_x_train,
                                                                                                   lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
        train_losses, val_losses, train_accs, val_accs, accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = train_GAT(
                                                                                                   fold2_x_test,
                                                                                                   fold2_x_train,
                                                                                                   lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
        train_losses, val_losses, train_accs, val_accs, accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = train_GAT(
                                                                                                   fold3_x_test,
                                                                                                   fold3_x_train,
                                                                                                   lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter)
        train_losses, val_losses, train_accs, val_accs, accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = train_GAT(
                                                                                                   fold4_x_test,
                                                                                                   fold4_x_train,
                                                                                                    lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
        test_accs.append((accs_0+accs_1+accs_2+accs_3+accs_4)/5)
        ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
        ACC2.append((acc2_0 + acc2_1 +acc2_2 + acc2_3 +acc2_4) / 5)
        PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
        PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
        REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
        REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
        F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
        F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    print("pheme:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                      "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                                sum(ACC2) / iterations, sum(PRE1) / iterations, sum(PRE2) /iterations,
                                                                                sum(REC1) / iterations, sum(REC2) / iterations, sum(F1) / iterations, sum(F2) / iterations))
