import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pickle
#from transformers import *
import json
from torch.utils.data import DataLoader


# global
label2id = {
            "rumor": 0,
            "non-rumor": 1,
            }



class UdGraphDataset(Dataset): 
    def __init__(self, fold_x, droprate): 
        
        self.fold_x = fold_x
        #self.data_path = data_path
        self.droprate = droprate

    def __len__(self): 
        return len(self.fold_x)

    def __getitem__(self, index): 
        id =self.fold_x[index] 
        
        # ====================================edgeindex==============================================
    
        with open('./data/pheme/all/'+ id + '/tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)
        #print(tweets)
        dict = {}
        for index, tweet in enumerate(tweets):
            dict[tweet] = index
        #print('dict: ', dict)

        with open('./data/pheme/all/'+ id + '/structure.pkl', 'rb') as f:
            inf = pickle.load(f)

        inf = inf[1:]

        #print(inf)
        # id to num
        new_inf = []
        for pair in inf:
            new_pair = []
            for E in pair:
                if E == 'ROOT':
                    break
                E = dict[E]
                new_pair.append(E)
            if E != 'ROOT':
                new_inf.append(new_pair)
        new_inf = np.array(new_inf).T
        edgeindex = new_inf
        #print('edgeindex: ', edgeindex.shape)
        #print(id)
    
        row = list(edgeindex[0]) 
        col = list(edgeindex[1]) 
        burow = list(edgeindex[1]) 
        bucol = list(edgeindex[0]) 
        row.extend(burow) 
        col.extend(bucol) 

        if self.droprate > 0: 
            length = len(row) 
            poslist = random.sample(range(length), int(length * (1 - self.droprate))) # 
            poslist = sorted(poslist)
            row1 = list(np.array(row)[poslist])
            col1 = list(np.array(col)[poslist])

            poslist2 = random.sample(range(length), int(length * (1 - self.droprate))) # 
            poslist2 = sorted(poslist2)
            row2 = list(np.array(row)[poslist2])
            col2 = list(np.array(col)[poslist2])

            new_edgeindex = [row1, col1] 
            new_edgeindex2 = [row2, col2]
        else:
            new_edgeindex = [row, col] 
            new_edgeindex2 = [row, col]

        

        # =========================================X=========================================================
        with open('./bert_w2c/PHEME/pheme_mask/' + id + '.json', 'r') as j_f:
            json_inf = json.load(j_f)
        
        x_list = json_inf[id]
        

        with open('./data/pheme/pheme_label.json', 'r') as j_tags:
            tags = json.load(j_tags)

        y = label2id[tags[id]]
        #y = np.array(y)
        if self.droprate > 0:
            #y = np.array(y)
            zero_list = [0]*768
            x_length = len(x_list)
            r_list = random.sample(range(x_length), int(x_length * self.droprate))
            r_list = sorted(r_list)
            for idex, line in enumerate(x_list):
                for r in r_list:
                    if idex == r:
                        x_list[idex] = zero_list
        
            x = np.array(x_list)
        else:
            x = np.array(x_list)

        return Data(x0=torch.tensor(x,dtype=torch.float32),
                x=torch.tensor(x,dtype=torch.float32), 
                edge_index=torch.LongTensor(new_edgeindex),
                edge_index2=torch.LongTensor(new_edgeindex2), 
                y1=torch.LongTensor([y]),
                y2=torch.LongTensor([y]),
                id=id) 



class test_UdGraphDataset(Dataset): 
    def __init__(self, fold_x, droprate): 
        
        
        self.fold_x = fold_x
        #self.data_path = data_path
        self.droprate = droprate

    def __len__(self): 
        return len(self.fold_x)

    def __getitem__(self, index): 
        id =self.fold_x[index] 
        
        # ====================================edgeindex==============================================
    
        with open('./data/pheme/all/'+ id + '/tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)
        #print(tweets)
        dict = {}
        for index, tweet in enumerate(tweets):
            dict[tweet] = index
        #print('dict: ', dict)

        with open('./data/pheme/all/'+ id + '/structure.pkl', 'rb') as f:
            inf = pickle.load(f)

        inf = inf[1:]

        #print(inf)
        # id to num
        new_inf = []
        for pair in inf:
            new_pair = []
            for E in pair:
                if E == 'ROOT':
                    break
                E = dict[E]
                new_pair.append(E)
            if E != 'ROOT':
                new_inf.append(new_pair)
        new_inf = np.array(new_inf).T
      
        edgeindex = new_inf
        #print('edgeindex: ', edgeindex.shape)
        #print(id)
       
        
        row = list(edgeindex[0])
        col = list(edgeindex[1]) 
        burow = list(edgeindex[1]) 
        bucol = list(edgeindex[0]) 
        row.extend(burow) 
        col.extend(bucol) 
        #print('new_edgeindex； ', np.array([row, col]).shape)
   
        if self.droprate > 0: 
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate))) # 
            poslist = sorted(poslist)
            row1 = list(np.array(row)[poslist])
            col1 = list(np.array(col)[poslist])

            poslist2 = random.sample(range(length), int(length * (1 - self.droprate))) # 
            poslist2 = sorted(poslist2)
            row2 = list(np.array(row)[poslist2])
            col2 = list(np.array(col)[poslist2])

            new_edgeindex = [row1, col1] 
            new_edgeindex2 = [row2, col2]
        else:
            new_edgeindex = [row, col] 
            new_edgeindex2 = [row, col]

     

        # =========================================X=========================================================
        with open('./bert_w2c/PHEME/pheme_mask/' + id + '.json', 'r') as j_f:
            json_inf = json.load(j_f)
        
        x = json_inf[id]
        x = np.array(x)

        with open('./data/pheme/pheme_label.json', 'r') as j_tags:
            tags = json.load(j_tags)

        y = label2id[tags[id]]
        #y = np.array(y)

        return Data(x0=torch.tensor(x,dtype=torch.float32),
                x=torch.tensor(x,dtype=torch.float32),
                edge_index=torch.LongTensor(new_edgeindex),
                edge_index2=torch.LongTensor(new_edgeindex2),  
                y1=torch.LongTensor([y]),
                y2=torch.LongTensor([y]),
                id=id)