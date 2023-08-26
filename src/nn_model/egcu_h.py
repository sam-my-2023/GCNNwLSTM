# model 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SimpleConv,GCNConv

from torch import Tensor
from torch.nn import Parameter as Param

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
    
class Summerize(nn.Module):
    def __init__(self, in_channel, shrink_k):
        super().__init__()
        self.k = shrink_k
        self.in_channel = in_channel
        self.p = Param(Tensor(in_channel))

    def reset_parameters(self):
        uniform(self.in_channel,self.p)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        p = self.p.repeat(batch_size,1).unsqueeze(2)
        y = torch.bmm(inputs,p)/torch.norm(self.p)
        _,top_k = torch.topk(y,self.k,dim=1)
        
        x = torch.mul(inputs,torch.tanh(y))
        x = torch.stack([x[i][top_k.squeeze(2)[i]] for i in range(batch_size)])
        return x

class EGCU_h(nn.Module):
    '''
    
    grcu: (node_n, in_channel) + ( in_channel, hidden_channel_1) -> (in_channel, hidden_channel_1)
    output from grcu w: (node_n ,in_channel) -> (node_n, hidden_channel_1)
    gcnn: (node_n, hidden_channel_1) + edge -> (node_n, out_channel)
    '''
    def __init__(self, in_channel,hidden_channel,out_channel, shrink_k, num_stocks):
        super(EGCU_h,self).__init__()

        # self.gcnn = SimpleConv(aggr= aggr_type)
        # self.shrink_layer = Summerize(shrink_k)
        self.gcnn = GCNConv(hidden_channel,out_channel,improved=True)
        self.reshape_layer = View([-1, in_channel,hidden_channel])
        self.graph_batch = View([-1,hidden_channel])
        self.graph_reverse = View([-1,num_stocks,out_channel])
        self.summer = Summerize(in_channel,shrink_k)
        self.gruc = torch.nn.GRUCell(shrink_k*in_channel, in_channel*hidden_channel)

        
    def forward(self, input, edge_index, last_graph_weights):
        x_ = self.summer(input)
        w = self.gruc(x_.flatten(start_dim=1),last_graph_weights.flatten(start_dim=1))
        w = self.reshape_layer(w)
        x = torch.bmm(input,w)
        x = x.flatten(end_dim=1)
        x = self.gcnn(x,edge_index)
        x = self.graph_reverse(x) 
        return x,w
    
def module_test():
    a = torch.randn(8,30,22)
    shrink_k = 10
    last_w = torch.ones(8,22,16)*0.5
    last_w_1 = torch.ones(8,20,16)*0.5

    edge_index = torch.stack([torch.tensor([0,1,2],dtype=torch.long),torch.tensor([1,2,1],dtype=torch.long)])
    edge_index = torch.cat([edge_index , edge_index .flip(dims=[0])],dim=1)
    egcu = EGCU_h(22,16,20,shrink_k,30)
    egcu_1 = EGCU_h(20,16,20,shrink_k,30)
    x,w = egcu(a,edge_index,last_w)
    x,w = egcu_1(x,edge_index,last_w_1)
    print(x)
    # print(w)
    
    
if __name__ == '__main__':
    module_test()
        