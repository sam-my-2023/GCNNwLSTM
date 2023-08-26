import pandas as pd
import numpy as np
import os
import torch
import sys, os
from source.nn_model import graph_nn
from torch_geometric.utils import negative_sampling
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Embedding(nn.Module):
    def __init__(self,in_channel, hidden_channel , out_channel,*args) -> None:
        super().__init__()
        self.in_nn = nn.Linear(in_channel, hidden_channel)
        self.rnn = nn.LSTM(hidden_channel,hidden_channel)
        # nn.Linear(hidden_channel,out_channel)
    
    def forward(self, inputs):
        x = self.in_nn(inputs)
        output, (hn, cn) = self.rnn(x)
        return output

class LSTM(nn.Module):
    def __init__(self,in_channel, hidden_channel , out_channel,*args) -> None:
        super().__init__()
        self.in_nn = nn.Linear(in_channel, hidden_channel)
        self.rnn = nn.LSTM(hidden_channel,hidden_channel)
        self.out_nn = nn.Sequential(
                            nn.Linear(hidden_channel, hidden_channel),
                            nn.ReLU(),
                            nn.Linear(hidden_channel, hidden_channel),
                            nn.ReLU(),
                            nn.Linear(hidden_channel, out_channel)
                        )
        # nn.Linear(hidden_channel,out_channel)
    
    def forward(self, inputs):
        x = self.in_nn(inputs)
        output, (hn, cn) = self.rnn(x)
        x = self.out_nn(hn)
        x = x.flatten()

        x = F.sigmoid(x)
        
        # x = x.permute(1,3,0,2)
        
        # x = F.log_softmax(x,dim=1)
        # x = x/torch.norm(x,dim=-1,keepdim=True)
        # corr = torch.stack([torch.stack([torch.matmul(x[i,j],x[i,j].T) for j in range(batch_size)])for i in range(seq_l)])
        # corr = torch.sum(x*x,axis =-1)/torch.norm(x)
        # corr = torch.tanh(corr)
        # x = torch.matmul(x,)
        
        return x