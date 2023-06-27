import torch as th
import torch.nn as nn
import torch.functional as F
from torch_scatter import scatter_mean
from Gnn.fc import *
from Att.att import *
from CL.Cl import *
from torch_geometric.nn import GCNConv,GATConv,GAE,GraphSAGE


class GAT_Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats,dropout = 0.2,batchsize = 240):
        super(GAT_Net, self).__init__()
        self.dropout = dropout
        #self.conv1 = GATConv(in_feats, hid_feats*2,heads=8,concat=False)
        self.conv1 = GATConv(in_feats, hid_feats * 2)
        self.conv2 = GATConv(hid_feats*2, out_feats)
        self.fc = th.nn.Linear(2 * out_feats, 4)
        self.hard_fc1 = hard_fc(out_feats, out_feats)
        self.hard_fc2 = hard_fc(out_feats, out_feats)  # optional
        self.hard_fc1_1 = hard_fc(out_feats,out_feats,DroPout=0.1)
        self.hard_fc1_2 = hard_fc(out_feats, out_feats, DroPout=0.2)
        self.hard_fc1_3 = hard_fc(out_feats, out_feats, DroPout=0.3)
        self.hard_fc1_4 = hard_fc(out_feats, out_feats, DroPout=0.4)
        self.batchsize = batchsize
        self.x_1 = 0
        self.x_2 = 0
        self.loss = DualLoss(0.5,0.1)
        self.gru =nn.GRU(input_size=in_feats, hidden_size=768)

        # 加入attention
        self.input_proj = nn.Linear(in_feats, out_feats, bias=False)
        self.output_proj = nn.Linear(in_feats, out_feats, bias=False)

    def forward(self, data):
        init_x0, init_x, edge_index1, edge_index2 = data.x0, data.x, data.edge_index, data.edge_index2

      
        att1 = attention3()
        init_x0, _ = att1.forward(init_x0, init_x0, init_x0)
        att2 = attention3()
        init_x, _ = att2.forward(init_x, init_x, init_x)
        init_x0 , gru_hiden = self.gru(init_x0)
        init_x, gru_hiden = self.gru(init_x)
        x1 = self.conv1(init_x0, edge_index1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1, edge_index1)
        x1 = F.elu(x1)
        x1 = scatter_mean(x1, data.batch, dim=0)
        x1_g = x1
        x1 = self.hard_fc1(x1)
        x1_t = x1
        x1 = th.cat((x1_g, x1_t), 1)

 
        x2 = self.conv1(init_x, edge_index2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = scatter_mean(x2, data.batch, dim=0)
        x2_g = x2
        x2 = self.hard_fc1(x2)
        x2_t = x2
        x2 = th.cat((x2_g, x2_t), 1)

        x = th.cat((x1, x2), 0)
        y = th.cat((data.y1, data.y2), 0)
        x_1 = x
        ################################fuyangben1##################################
        x1 = self.conv1(init_x0, edge_index1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1, edge_index1)
        x1 = F.relu(x1)
        x1 = scatter_mean(x1, data.batch, dim=0)
        x1_g = x1
        x1 = self.hard_fc1_1(x1)
        x1_t = x1
        x1 = th.cat((x1_g, x1_t), 1)

        x2 = self.conv1(init_x, edge_index2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2, edge_index2)
        x2 = F.relu(x2)
        x2 = scatter_mean(x2, data.batch, dim=0)
        x2_g = x2
        x2 = self.hard_fc1_1(x2)
        x2_t = x2
        x2 = th.cat((x2_g, x2_t), 1)

        x_1 = th.cat((x2, x1), 0)
        y_1 = th.cat((data.y2, data.y1), 0)
        ##################################fuyangben1##################################
        ################################fuyangben2##################################
        x1 = self.conv1(init_x0, edge_index1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1, edge_index1)
        x1 = F.relu(x1)
        x1 = scatter_mean(x1, data.batch, dim=0)
        x1_g = x1
        x1 = self.hard_fc1_2(x1)
        x1_t = x1
        x1 = th.cat((x1_g, x1_t), 1)

        x2 = self.conv1(init_x, edge_index2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2, edge_index2)
        x2 = F.relu(x2)
        x2 = scatter_mean(x2, data.batch, dim=0)
        x2_g = x2
        x2 = self.hard_fc1_2(x2)
        x2_t = x2
        x2 = th.cat((x2_g, x2_t), 1)

        x_2 = th.cat((x2, x1), 0)
        y_2 = th.cat((data.y2, data.y1), 0)
        ##################################fuyangben2##################################
        ################################fuyangben3##################################
        x1 = self.conv1(init_x0, edge_index1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1, edge_index1)
        x1 = F.relu(x1)
        x1 = scatter_mean(x1, data.batch, dim=0)
        x1_g = x1
        x1 = self.hard_fc1_3(x1)
        x1_t = x1
        x1 = th.cat((x1_g, x1_t), 1)

        x2 = self.conv1(init_x, edge_index2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2, edge_index2)
        x2 = F.relu(x2)
        x2 = scatter_mean(x2, data.batch, dim=0)
        x2_g = x2
        x2 = self.hard_fc1_3(x2)
        x2_t = x2
        x2 = th.cat((x2_g, x2_t), 1)

        x_3 = th.cat((x2, x1), 0)
        y_3 = th.cat((data.y2, data.y1), 0)
        ##################################fuyangben3##################################
        ################################fuyangben4##################################
        x1 = self.conv1(init_x0, edge_index1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1, edge_index1)
        x1 = F.relu(x1)
        x1 = scatter_mean(x1, data.batch, dim=0)
        x1_g = x1
        x1 = self.hard_fc1_4(x1)
        x1_t = x1
        x1 = th.cat((x1_g, x1_t), 1)

        x2 = self.conv1(init_x, edge_index2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2, edge_index2)
        x2 = F.relu(x2)
        x2 = scatter_mean(x2, data.batch, dim=0)
        x2_g = x2
        x2 = self.hard_fc1_4(x2)
        x2_t = x2
        x2 = th.cat((x2_g, x2_t), 1)

        x_4 = th.cat((x2, x1), 0)
        y_4 = th.cat((data.y2, data.y1), 0)
        ##################################fuyangben4##################################


        xs = x
        x_n = th.cat((x_1, x_2, x_3, x_4), 0)
        xx = x_n.view(-1, 4, 128)
        #####################################替换爱因斯坦求和######################################
        xs = torch.unsqueeze(xs, dim=1)
        predicts = (xs * xx).sum(2)
        xs = torch.squeeze(xs, dim=1)
        #####################################替换爱因斯坦求和######################################
        #predicts = th.einsum('bd,bcd->bc', xs, xx)
        outputs = {
            'predicts': predicts,
            'cls_feats': xs,
            'label_feats': xx
        }
        targets = y
        cl_loss = self.loss(outputs, targets)


        self.x_1 = x.size(0)
        self.x_2 = x.size(1)
        x = x.unsqueeze(0)
        avg = nn.AdaptiveAvgPool2d((self.x_1, self.x_2))
        x = avg(x)
        x = x.squeeze(0)

        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x, cl_loss, y


