import os
#from Process.dataset_pheme import UdGraphDataset, test_UdGraphDataset
from Process.dataset_pheme import UdGraphDataset, test_UdGraphDataset


cwd=os.getcwd()


def loadUdData(dataname, fold_x_train,fold_x_test,droprate):
    
    #print("loading train set", )
    traindata_list = UdGraphDataset(fold_x_train, droprate=droprate)
    #print("train no:", len(traindata_list))
    #print("loading test set", )
    testdata_list = test_UdGraphDataset(fold_x_test, droprate=0) # droprate*****
    #print("test no:", len(testdata_list))
    return traindata_list, testdata_list
