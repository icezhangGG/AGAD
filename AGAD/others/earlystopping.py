import numpy as np
import torch

class EarlyStopping:

    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs=0
        self.F1=0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.val_loss_min = np.Inf
        self.save_checkpoint = torch.save


    def __call__(self, val_loss, accs,F1,F2,F3,F4,model,modelname,str):

        score = (accs+F1+F2+F3+F4) /5

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.3f}|NR F1: {:.3f}|FR F1: {:.3f}|TR F1: {:.3f}|UR F1: {:.3f}"
                      .format(self.accs,self.F1,self.F2,self.F3,self.F4))
        else:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.save_checkpoint(model.state_dict(), './data/save.pth.tar')
            self.counter = 0

class EarlyStopping2:

    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.temp_val_Acc_all = 0
        self.temp_val_Acc1 = 0
        self.temp_val_Acc2 = 0
        self.temp_val_Prec1 = 0
        self.temp_val_Prec2 = 0
        self.temp_val_Recll1 = 0
        self.temp_val_Recll2 = 0
        self.temp_val_F1 = 0
        self.temp_val_F2 = 0

        self.val_loss_min = np.Inf
        self.save_checkpoint = torch.save

    def __call__(self, temp_val_losses, temp_val_Acc_all, temp_val_Acc1, temp_val_Acc2
                 ,temp_val_Prec1,temp_val_Prec2,temp_val_Recll1,temp_val_Recll2
                    ,temp_val_F1,temp_val_F2, model, modelname):

        score = (temp_val_Acc_all + temp_val_F1 + temp_val_F2) / 3

        if self.best_score is None:
            self.best_score = score
            self.temp_val_Acc_all = temp_val_Acc_all
            self.temp_val_Acc1=temp_val_Acc1
            self.temp_val_Acc2 = temp_val_Acc2
            self.temp_val_Prec1=temp_val_Prec1
            self.temp_val_Prec2 = temp_val_Prec2
            self.temp_val_Recll1 = temp_val_Recll1
            self.temp_val_Recll2 = temp_val_Recll2
            self.temp_val_F1 = temp_val_F1
            self.temp_val_F2 = temp_val_F2

        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                # print("BEST Accuracy: {:.3f}|NR F1: {:.3f}|FR F1: {:.3f}|TR F1: {:.3f}|UR F1: {:.3f}"
                #       .format(self.accs, self.F1, self.F2, self.F3, self.F4))
                print("BEST Accuracy:{:.3f}|N_Pre:{:.3f}|N_Rec:{:.3f}|N_F1:{:.3f}|R_Pre:{:.3f}|R_Rec:{:.3f}|R_F1:{:.3f}"
                      .format(self.temp_val_Acc_all,self.temp_val_Prec1,self.temp_val_Recll1,self.temp_val_F1,self.temp_val_Prec2,self.temp_val_Recll2,self.temp_val_F2))
        else:
            self.best_score = score
            self.temp_val_Acc_all = temp_val_Acc_all
            self.temp_val_Acc1 = temp_val_Acc1
            self.temp_val_Acc2 = temp_val_Acc2
            self.temp_val_Prec1 = temp_val_Prec1
            self.temp_val_Prec2 = temp_val_Prec2
            self.temp_val_Recll1 = temp_val_Recll1
            self.temp_val_Recll2 = temp_val_Recll2
            self.temp_val_F1 = temp_val_F1
            self.temp_val_F2 = temp_val_F2

            self.save_checkpoint(model.state_dict(), './data/save.pth.tar')
            self.counter = 0