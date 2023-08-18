import torch as th
import torch.nn as nn
import torch.functional as F
from torch_scatter import scatter_mean
from Gnn.fc import *
from Att.att import *
from CL.Cl import *
from torch_geometric.nn import GCNConv,GATConv,GAE,GraphSAGE


class SGAT_Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats,dropout = 0.2,batchsize = 240):
        super(SGAT_Net, self).__init__()
        self.dropout = dropout
        #self.conv1 = GATConv(in_feats, hid_feats*2,heads=8,concat=False)
        self.conv11 = GATConv(in_feats, hid_feats * 2,heads =8,concat=True)
        self.conv22 = GATConv(hid_feats*2, out_feats,heads=8,concat=False)
        self.conv1 = GATConv(in_feats, hid_feats * 2, heads=8, concat=False)
        self.conv2 = GATConv(hid_feats * 2, out_feats, heads=8, concat=False)
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
        #print("init_x0:", init_x0.shape)
        #print("init_x:", init_x.shape)
        init_x0 = init_x0.unsqueeze(0)
        init_x0 , gru_hiden = self.gru(init_x0)
        #print("gru_out_init_x0",init_x0.shape)
        init_x = init_x.unsqueeze(0)
        init_x, gru_hiden = self.gru(init_x)
   
        init_x0 = init_x0.squeeze(0)
        init_x = init_x.squeeze(0)
        x1 = self.conv11(init_x0, edge_index1)
        #############################平均池化层压缩维度##############################
        po = nn.AvgPool1d(8,8)
        x1 = po(x1.unsqueeze(1))
        x1 = x1.squeeze(1)
        ###########################################################
        ##############################线性层压缩维度#############################
        #lin = nn.Linear(x1.size(1), 128, bias=True).cuda()
        #x1 = lin(x1)
        ###########################################################
        #注意力机制
        x1 = F.relu(x1)
        #x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x1 = self.conv22(x1, edge_index1)
        #############################平均池化层压缩维度##############################
        # po = nn.AvgPool1d(8, 8)
        # x1 = po(x1.unsqueeze(1))
        # x1 = x1.squeeze(1)
        #x1 = lin(x1)
        #############################平均池化层压缩维度##############################
        x1 = F.elu(x1)
        x1 = scatter_mean(x1, data.batch, dim=0)
        x1_g = x1
        x1 = self.hard_fc1(x1)
        x1_t = x1
        x1 = th.cat((x1_g, x1_t), 1)
        x2 = self.conv11(init_x, edge_index2)
        #############################平均池化压缩维度############################
        po = nn.AvgPool1d(8, 8)
        x2 = po(x2.unsqueeze(1))
        x2 = x2.squeeze(1)
        #########################################################
        #############################线性层压缩维度############################
        #lin = nn.Linear(x2.size(1), 128, bias=True).cuda()
        #x2 = lin(x2)
        #########################################################
        x2 = F.relu(x2)
        #x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = self.conv22(x2, edge_index2)
        #############################平均池化层压缩维度##############################
        # po = nn.AvgPool1d(8, 8)
        # x2 = po(x2.unsqueeze(1))
        # x2 = x2.squeeze(1)
        #############################平均池化层压缩维度##############################
        x2 = F.elu(x2)
        x2 = scatter_mean(x2, data.batch, dim=0)
        x2_g = x2
        x2 = self.hard_fc1(x2)
        x2_t = x2
        x2 = th.cat((x2_g, x2_t), 1)
        ####################################attention########################################
        # att = AttentionLayer(x1.size(1),x1.size(1))
        # print("查看x1的维度:",x1.size())
        # print("查看x2的维度:", x2.size())
        # x1 = att.forward(x1,x2)
        ####################################attention########################################
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

        # x_T = x.t()
        # dot_matrix = th.mm(x, x_T)
        # x_norm = th.norm(x, p=2, dim=1)
        # x_norm = x_norm.unsqueeze(1)
        # norm_matrix = th.mm(x_norm, x_norm.t())
        # t = 0.3
        # cos_matrix = (dot_matrix / norm_matrix) / t
        # cos_matrix = th.exp(cos_matrix)
        # diag = th.diag(cos_matrix)
        # cos_matrix_diag = th.diag_embed(diag)
        # cos_matrix = cos_matrix - cos_matrix_diag
        # y_matrix_T = y.expand(len(y), len(y))
        # y_matrix = y_matrix_T.t()
        # y_matrix = th.ne(y_matrix, y_matrix_T).float()
        #
        # neg_matrix = cos_matrix * y_matrix
        # neg_matrix_list = neg_matrix.chunk(2, dim=0)
        # pos_y_matrix = y_matrix * (-1) + 1
        # pos_matrix_list = (cos_matrix * pos_y_matrix).chunk(2, dim=0)
        # # print('cos_matrix: ', cos_matrix.shape, cos_matrix)
        # # print('pos_y_matrix: ', pos_y_matrix.shape, pos_y_matrix)
        # pos_matrix = pos_matrix_list[0]
        # # print('pos shape: ', pos_matrix.shape, pos_matrix)
        # neg_matrix = (th.sum(neg_matrix, dim=1)).unsqueeze(1)
        # sum_neg_matrix_list = neg_matrix.chunk(2, dim=0)
        # p1_neg_matrix = sum_neg_matrix_list[0]
        # p2_neg_matrix = sum_neg_matrix_list[1]
        # neg_matrix = p1_neg_matrix
        # # print('neg shape: ', neg_matrix.shape)
        # div = pos_matrix / neg_matrix
        # div = (th.sum(div, dim=1)).unsqueeze(1)
        # div = div / self.batchsize
        # log = th.log(div)
        # SUM = th.sum(log)
        # cl_loss = -SUM

        self.x_1 = x.size(0)
        self.x_2 = x.size(1)
        x = x.unsqueeze(0)
        avg = nn.AdaptiveMaxPool2d((self.x_1, self.x_2))
        x = avg(x)
        x = x.squeeze(0)

        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x, cl_loss, y
