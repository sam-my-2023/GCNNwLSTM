# model 
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
    
class GCNN_RNN_Hidden(nn.Module):
    '''
    this neural network use GCNN to propagate the hidden cells in RNN
    '''
    def __init__(self, input_channel,hidden_channel,to_center,to_local, num_stocks, num_news):
        super(GCNN_RNN_Hidden,self).__init__()
        self.num_stocks = num_stocks
        self.num_news = num_news
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.stock_to_news_propogation = SimpleConv(aggr= to_center)
        self.news_to_stock_propogation = SimpleConv(aggr= to_local)
        
        # ! should each company have the same begining hidden state or different
        # here we use the same beginning state
        self.h_start = Param(Tensor(hidden_channel))
        self.h_input_start = Param(Tensor(hidden_channel))
        self.m_c_start = Param(Tensor(hidden_channel))
        # self.center_layer = nn.Sequential(
        #                     nn.Linear( input_channel, hidden_channel),
        #                     nn.ReLU(),
        #                     nn.Linear(hidden_channel, hidden_channel))
        self.center_layer = torch.nn.GRUCell(input_channel, hidden_channel)
        self.local_rnn_hiden = torch.nn.GRUCell(input_channel+hidden_channel, hidden_channel)
        self.local_rnn_input = torch.nn.GRUCell(input_channel, hidden_channel)
        
        
        
        # self.output_MLP = nn.Sequential(
        #                     nn.Linear(embedding+ stock_lstm_para[1], 64),
        #                     nn.ReLU(),
        #                     nn.Linear(64, class_size),
        #                     nn.Tanh()
        #                 )
        
        # self.reset_parameters()
    
    def reset_parameters(self):
        self.stock_to_news_propogation.reset_parameters()
        self.news_to_stock_propogation.reset_parameters()
        uniform(self.hidden_channel,self.h_start)
        uniform(self.hidden_channel,self.h_input_start)
        uniform(self.hidden_channel,self.m_c_start)
        self.center_layer.reset_parameters()
        self.local_rnn_hiden.reset_parameters()
        self.local_rnn_input.reset_parameters()
        
    def forward(self, inputs, edge_index):
        seq_l = inputs.size(0)
        stocks_len = inputs.size(1)
        batch_size = int(stocks_len/self.num_stocks)
        news_len = batch_size*self.num_news
        channel = inputs.size(2)
        
        # initialize the hidden state
        # h = [self.h_start.repeat(stocks_len,1)]
        h = self.h_start.repeat(stocks_len,1)
        h_input = self.h_input_start.repeat(stocks_len,1)
        m_c = self.m_c_start.repeat(news_len,1)
        
        back_edge = torch.flip(edge_index,dims=[1])
        # message propagation
        for i in range(seq_l):
            center_m = self.stock_to_news_propogation([inputs[i],torch.empty(news_len,channel)],back_edge[i])
            # center_m = self.center_layer(center_m)
            m_c = self.center_layer(center_m, m_c)
            center_m = torch.cat([center_m,m_c],dim=-1)
            m = self.news_to_stock_propogation([center_m,torch.empty(stocks_len,channel+self.hidden_channel)], edge_index[i])
            
            # m = torch.matmul(m, self.weight[2*j+1])
            # m = torch.cat([inputs[i],m], dim=-1)
            h = self.local_rnn_hiden(m, h)
            h_input = self.local_rnn_input(inputs[i],h_input)
            # h.append(self.local_rnn(m, h[-1]))
        return m,h,h_input
        # return torch.stack(h,dim=0)[1:].reshape(seq_l,batch_size,self.num_stocks,self.hidden_channel)

class Our_Model(nn.Module):
    def __init__(self,in_channel, hidden_channel , out_channel, args_1, args_2) -> None:
        super().__init__()
        self.in_nn = nn.Sequential(
                            nn.Linear(in_channel, hidden_channel),
                            nn.ReLU(),
                            nn.Linear(hidden_channel, hidden_channel),
                            nn.ReLU(),
                            nn.Linear(hidden_channel, hidden_channel)
                        )
        chanels = 0
        self.rnn_gccn_input = GCNN_RNN_Hidden(in_channel,hidden_channel,*args_1,*args_2)
        # chanels += 4*hidden_channel
        self.rnn_gccn_hidden = GCNN_RNN_Hidden(hidden_channel,hidden_channel,*args_1,*args_2)
        # chanels += in_channel+3*hidden_channel
        self.out_nn = nn.Sequential(
                            nn.Linear(hidden_channel, 4*hidden_channel),
                            nn.ReLU(),
                            nn.Linear(4*hidden_channel, 2*hidden_channel),
                            nn.ReLU(),
                            nn.Linear(2*hidden_channel, hidden_channel),
                            nn.ReLU(),
                            nn.Linear(hidden_channel, out_channel)
                        )
        # nn.Linear(hidden_channel,out_channel)
    
    def forward(self, inputs, edge_index):
        x = self.in_nn(inputs)
        m,h,h_input = self.rnn_gccn_hidden(x,edge_index)
        m_i,h_i,h_input_i = self.rnn_gccn_input(inputs,edge_index)
        x = torch.cat([m,h,h_input],dim=-1)
        y = torch.cat([m_i,h_i,h_input_i],dim=-1)
        # x = torch.cat([x,y],dim=-1)
        x = self.out_nn(h_input)
        # seq_l, batch_size,num_stocks, embedding_dim = x.size()
        # print(x.size())
        x = x.squeeze()
        x = F.sigmoid(x)
        
        # x = x.permute(1,3,0,2)
        
        # x = F.log_softmax(x,dim=1)
        # x = x/torch.norm(x,dim=-1,keepdim=True)
        # corr = torch.stack([torch.stack([torch.matmul(x[i,j],x[i,j].T) for j in range(batch_size)])for i in range(seq_l)])
        # corr = torch.sum(x*x,axis =-1)/torch.norm(x)
        # corr = torch.tanh(corr)
        # x = torch.matmul(x,)
        
        return x