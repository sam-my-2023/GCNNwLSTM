import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SimpleConv

from torch import Tensor
from torch.nn import Parameter as Param

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm
    
class GCNN(nn.Module):
    '''
    this neural network use GCNN to propagate the hidden cells in RNN
    '''
    def __init__(self, ignore_in_channel,in_channel,hidden_channel,out_channel,to_center,to_local, num_stocks, num_news):
        super(GCNN,self).__init__()
        self.num_stocks = num_stocks
        self.num_news = num_news
        if ignore_in_channel:
            self.in_channel = out_channel
        else:
            self.in_channel = in_channel
        self.out_channel = out_channel
        self.in_layer = nn.Sequential(
                            nn.Linear( self.in_channel, hidden_channel),
                            nn.ReLU(),
                            )
        
        self.stock_to_news_propogation = SimpleConv(aggr= to_center)
        self.mid_layer = nn.Sequential(
                            nn.Linear( hidden_channel, hidden_channel),
                            nn.ReLU(),
                            )
        self.news_to_stock_propogation = SimpleConv(aggr= to_local)
        self.out_layer = nn.Sequential(
                            nn.Linear( hidden_channel, self.out_channel),
                            nn.ReLU(),
                            )
    
        
    def forward(self, inputs, edge_index):
        stocks_len = inputs.size(0)
        batch_size = int(stocks_len/self.num_stocks)
        news_len = batch_size*self.num_news
        
        back_edge = torch.flip(edge_index,dims=[0])
        # message propagation
        m = self.in_layer(inputs)
        center_m = self.stock_to_news_propogation([m,torch.empty(news_len,self.in_channel)],back_edge)
            
        center_m = self.mid_layer(center_m)
        
        m = self.news_to_stock_propogation([center_m,torch.empty(stocks_len,self.out_channel)], edge_index)
        m = self.out_layer(m)
        return m

class Graph_Embedding(nn.Module):
    def __init__(self, n_layers,**kwarg):
        super(Graph_Embedding,self).__init__()
        self.gcnn_first = GCNN(ignore_in_channel=False, **kwarg)
        self.gcnn_layers = GCNN(ignore_in_channel=True, **kwarg)
    
    def forward(self, inputs, edge_index):
        
        x = self.gcnn_first(inputs, edge_index)
        x = self.gcnn_layers(x, edge_index)
        return x
    
class Our_Model(nn.Module):
    def __init__(self, n_layers, n_class, **kwarg):
        super(Our_Model,self).__init__()
        self.embedding = Graph_Embedding(n_layers,**kwarg)
        self.out = nn.Linear(kwarg['out_channel'],n_class)
    
    def forward(self, inputs, edge_index):
        x = self.embedding(inputs, edge_index)
        x = self.out(x)
        x = F.sigmoid(x)
        return x