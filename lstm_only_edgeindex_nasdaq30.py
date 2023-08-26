import os
import sys
PROJECT_PATH = os.path.abspath('..')
sys.path.append(PROJECT_PATH)

from collections import defaultdict
import json
import pandas as pd

import numpy as np

# time series data modules
from src.data_process import graph_time_series, stock_time_series 

# nn modules
from torch.utils.data import DataLoader

import torch
from src.nn_model import graph_nn,egcu_h
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


from sklearn.metrics import f1_score,accuracy_score

class From_Jason_File:
    '''
    a wrapper for data read from chatgpt jason
    '''
    def __init__(self, nest_list_data) -> None:
        
        # dtype as np.int64 corresponds to pandas df wrds 
        self.dates = [np.int64(e['Date'].replace('-', '')) for e in nest_list_data]
        
        self.affected_companies = [e['Affected Companies'] for e in nest_list_data]
        self._process()
        
    def _process(self):
        total_companies =  [company for daily_afftected in self.affected_companies for company in list(daily_afftected.keys())]
        self.total_companies = list(set(total_companies))
        num_dates = len(self.dates)
        relation = []
        for i in range(num_dates):
            tem = defaultdict(lambda: [])
            for company,flag in self.affected_companies[i].items():
                tem[flag].append(company)
            relation.append(tem)
            
        
        self.relation = relation

    def __len__(self):
        return len(self.dates)
    
    def get_graphs(self,  relation_type = "positive"):
        return [x[relation_type] for x in self.relation]
    
    def get_dates(self):
        return self.dates
    
    def get_graph_data(self, date_idx=0, date = None, relation_type = "positive"):
        if date is not None:
            print(date)
        else:
            return self.relation[date_idx][relation_type]

def edge_index_single_day(positive_rel,negative_rel,TICKERS_IND):

    m = len(TICKERS_IND)

 # positive
    pos_edge = []
    for graph in positive_rel:
        tem = []
        for k in range(len(graph)):
            i = TICKERS_IND[graph[k]]
            for c in graph[k+1:]:
                j = TICKERS_IND[c]
                tem.append([i,j])
                tem.append([j,i])
        pos_edge.append(np.array(tem))

 # negative set
    neg_edge = []
    
    for graph in negative_rel:
        tem = []
        for k in range(len(graph)):
            i = TICKERS_IND[graph[k]]
            for c in graph[k+1:]:
                j = TICKERS_IND[c]
                tem.append([i,j])
                tem.append([j,i])
        neg_edge.append(np.array(tem))

    
    # negative corr
 #    
    neg_corr_edge = []
    
    for pg,ng in list(zip(positive_rel,negative_rel)):
        tem = []
        for c_p in pg:
            i = TICKERS_IND[c_p]
            for c_n in ng:
                j = TICKERS_IND[c_n]
                if i!=j and [i,j] not in tem:
                    tem.append([i,j])
                    tem.append([j,i])
        neg_corr_edge.append(np.array(tem))
                    

    return pos_edge,neg_edge,neg_corr_edge

class Our_Dataset:
    def __init__(self,data_per_date_per_company,tag_per_date_per_company, egde_per_date, l, mask_size, recurrent_training = True):
        
        self.n = min(len(egde_per_date),                    
                     data_per_date_per_company.shape[0],
                     tag_per_date_per_company.shape[0])
        
        self.num_companies = data_per_date_per_company.shape[1]
        assert self.num_companies == tag_per_date_per_company.shape[1]
        self.egde_per_date = egde_per_date
        self.data_per_date_per_company = data_per_date_per_company
        self.tag_per_date_per_company = tag_per_date_per_company
        self.l = l
        self.mask_size = mask_size
        self.mask_number = self.num_companies//mask_size
        assert self.mask_number*mask_size==self.num_companies
        
        rng = np.random.default_rng()
        tem = np.vstack(( np.arange(self.num_companies), ) *self.n)
        self.random_select_per_day = rng.permuted(tem, axis=1)
        
        self.recurrent_training = recurrent_training
    
    def __getitem__(self,i):
        j = i//self.mask_number
        k = i%self.mask_number
        select = self.random_select_per_day[j,k*self.mask_size:(k+1)*self.mask_size]
        assert j<(self.n-self.l)
        if self.recurrent_training:
            tag = self.tag_per_date_per_company[j:j+self.l,select]
        else:
            tag = [self.tag_per_date_per_company[j+self.l,select]]
        return self.data_per_date_per_company[j:j+self.l,:], tag, self.egde_per_date[j:j+self.l], select
    
    def __len__(self):
        return (self.n-self.l)*self.mask_number
    

def our_collate_fn(batch):
    inps = torch.tensor(np.concatenate([x[0] for x in batch],axis=1),dtype=torch.float32)
    inps = inps.permute(1,0,2)

    
    corr = torch.tensor(np.concatenate([x[1] for x in batch],axis=1), dtype=torch.long)
    corr = corr.flatten()
    

    batch_n = inps.size(0)
    seq_m = inps.size(1)
    num_stocks = inps.size(2)
    
    adj_ms = torch.empty(10)
    # adj_ms =torch.tensor(np.stack([x[2] for x in batch],axis=0).T, dtype=torch.long) 
    
    batch_select = torch.tensor(np.concatenate([ i*num_stocks+x[3] for i,x in enumerate(batch)]),dtype=torch.long)
    
    return inps, corr, adj_ms, batch_select

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
        
class Our_Model(nn.Module):
    '''
        flatten:

        GCNN: (batch_size*sequence_length*num_stocks,in_channel) + edge_idx -> (batch_size*sequence_length*num_stocks, graph_channel) 
        
        unflatten: (batch_size*sequence_length*num_stocks, graph_channel) -> (batch_size, sequence_length, graph_channel)

        Cat: (batch_size, sequence_length, graph_channel) + ((batch_size, sequence_length, feature_size) -> (~,~, graph_channel+ feature_size)

        lstm (w/o outputing hidden state): (batch_size, sequence_length, lstm_in_channel)  -> (batch, out_channel)

    '''
    def __init__(self, in_channel, graph_channel, lstm_out_channel, out_channel,num_lstm_layer,input_MLP=None, hidden_channels=None, recurrent_training = True):
    
        super(Our_Model,self).__init__()
        # self.gcnn = GCNConv(in_channel, graph_channel)
        
        if input_MLP:
            input_MLP_list = [nn.Linear(in_channel, input_MLP[0][0])]
            if input_MLP[0][1]:
                input_MLP_list.append(input_MLP[0][1])
            
            for k in range(1,len(input_MLP)):
                input_MLP_list.append(nn.Linear(input_MLP[k-1][0], input_MLP[k][0]))
                if input_MLP[k][1]:
                    input_MLP_list.append(input_MLP[k][1])
            in_channel = input_MLP[-1][0]
            self.input_mlp = nn.Sequential(*input_MLP_list)
        else:
            self.input_mlp = nn.Identity()
            
        
        
        if lstm_out_channel[0]:
            self.comb_lstm = nn.LSTM(graph_channel + in_channel, lstm_out_channel[0], batch_first = True)
        
        
        self.stock_lstm = nn.LSTM(in_channel, lstm_out_channel[1], batch_first = True, num_layers = num_lstm_layer[1])
        
        
        self.recurrent_training = recurrent_training
        if not recurrent_training:
            lstm_out = num_lstm_layer[1]*lstm_out_channel[1]
        else:
            lstm_out = lstm_out_channel[1]
        if hidden_channels:
            final_MLP_list = [nn.Linear(lstm_out, hidden_channels[0][0])]
            if hidden_channels[0][1]:
                final_MLP_list.append(hidden_channels[0][1])
            
            for k in range(1,len(hidden_channels)):
                final_MLP_list.append(nn.Linear(hidden_channels[k-1][0], hidden_channels[k][0]))
                if hidden_channels[k][1]:
                    final_MLP_list.append(hidden_channels[k][1])
            
            final_MLP_list.append(nn.Linear(hidden_channels[-1][0], out_channel))
            
        else:
            final_MLP_list = [nn.Linear(lstm_out, out_channel)]
        
        self.output_MLP = nn.Sequential(*final_MLP_list)
        
    def forward(self, inputs, edge_index, select):
        '''
        inputs: (batch_size, seq_length, num_stocks, feature_size)
    
        '''
        data_size = inputs.size()
        # x = inputs.flatten(start_dim=0, end_dim= -2)
        # x_graph = self.gcnn(x,edge_index)
        # x_graph = torch.concat((x_graph,x), dim=1)
        # x_graph = x_graph.unflatten(dim=0, sizes=data_size[:-1] )
        inputs = self.input_mlp(inputs)
        stock_input = inputs[select]
        # x_graph = x_graph.permute(0,2,1,3)

        # data_size = inputs.size()

        # comb_o, (comb_h,comb_c) = self.comb_lstm(x_graph.flatten(start_dim=0,end_dim=1)) 
        stock_o, (stock_h,stock_c) = self.stock_lstm(stock_input)
        if self.recurrent_training:
            output = stock_o
        else:
            output = stock_h.permute((1,0,2))
            output = output.flatten(start_dim=1)
        # print(stock_hidden.size())
        # output = torch.concat((stock_h[-1],comb_h[-1]),dim=1)
        # output = self.output_MLP(stock_h[-1])
        output = self.output_MLP(output)
        if self.recurrent_training:
            output = output.permute(1,0,2)
            output = output.flatten(end_dim=1)
        return F.log_softmax(output,dim=1)



if __name__=='__main__':
    with open(PROJECT_PATH + "/data/ticker_train_data.json") as json_file:
        ticker_train_data = json.load(json_file)
    json_wrapper = From_Jason_File(ticker_train_data)

    with open(PROJECT_PATH + "/data/ticker_test_data.json") as json_file:
        ticker_test_data = json.load(json_file)
    test_json_wrapper = From_Jason_File(ticker_test_data)

    TICKERS = ['HD', 'INTC', 'JPM', 'CAT', 'WMT', 'HON', 'JNJ', 'AXP', 'DIS', 'UNH', 'CVX', 'PG', 'NKE', 'IBM', 'CRM', 'AMGN', 'KO', 'BA', 'CSCO', 'TRV', 'AAPL', 'MSFT', 'DOW', 'MRK', 'MMM', 'WBA', 'V', 'GS', 'MCD', 'VZ']
    TICKERS_IDX = {ticker:i for i,ticker in enumerate(TICKERS)}
    
    price = stock_time_series.get_stock_price(TICKERS, os.path.join(PROJECT_PATH,'data'))
    price = price.squeeze()
    stock_feature = stock_time_series.get_stock_feature(TICKERS, os.path.join(PROJECT_PATH,'data'))
    
    # stock_dates = stock_time_series.get_dates(TICKERS, os.path.join(PROJECT_PATH,'data'))
    # pos_dates = json_wrapper.get_dates()
    # test_dates = test_json_wrapper.get_dates()
    # print(stock_dates[0], pos_dates[0], stock_dates[len(pos_dates)],test_dates[0])
    
    tag = np.stack([1*((price[i+1]/price[i])>=1.01) + 1*((price[i+1]/price[i])>.99) for i in range(price.shape[0]-1)])


    positive_rel = json_wrapper.get_graphs()
    negative_rel = json_wrapper.get_graphs(relation_type='negative')
    pos_edge,neg_edge,neg_corr_edge = edge_index_single_day(positive_rel, negative_rel,TICKERS_IDX)

    n = min(price.shape[0],stock_feature.shape[0], len(positive_rel))
    m = len(TICKERS)
    
    
    
    positive_rel = test_json_wrapper.get_graphs()
    negative_rel = test_json_wrapper.get_graphs(relation_type='negative')
    test_pos_edge,test_neg_edge,test_neg_corr_edge = edge_index_single_day(positive_rel, negative_rel,TICKERS_IDX)
    
    n_test = len(pos_edge)
    sep_point = n
    n_test = min(price.shape[0]-sep_point, stock_feature.shape[0]-n_test,n_test)
    end_point = n_test+sep_point
    
    
    l = 60
    batch_size = 16
    mask_size = 5
    recurent = False
    
    assert sep_point==len(pos_edge)

    our_data_set = Our_Dataset(stock_feature[:sep_point], tag[:sep_point], pos_edge[:sep_point], l,mask_size, recurrent_training=recurent)
    dataloader = DataLoader(our_data_set, batch_size, collate_fn = our_collate_fn, shuffle= True)
    

    test_ds = Our_Dataset(stock_feature[sep_point:end_point], tag[sep_point:end_point], test_pos_edge[:n_test], l, mask_size,recurrent_training=recurent)
    test_dl = DataLoader(test_ds, batch_size, collate_fn = our_collate_fn, shuffle= False)
    
    device = 'cuda'
    in_MLP = None #[[128,None],[128,None]]
    out_MLP = [[512,None],[128,None],[128,None]]
    our_model = Our_Model(22,128,[None,512],3,[None,4],in_MLP ,out_MLP,recurrent_training=recurent).to(device)
    optimizer = optim.RMSprop(our_model.parameters(), lr = 1e-2, momentum= 0.1)
    # optimizer = optim.Adam(our_model.parameters(), lr=1e-6,momentum=0.9)
    # optimizer = optim.SGD(our_model.parameters(), lr = 1e-7,momentum=0.1)
    # optimizer = optim.Adadelta(our_model.parameters())
    
    loss_fn = nn.NLLLoss()
 

    for ep in range(20):
        preds = []
        rets = []
        total_loss = 0
        for stock_sequence, ret_mean_batch, edge_sequence, batch_select in  dataloader:
            stock_sequence = stock_sequence.to(device)
            ret_mean_batch = ret_mean_batch.to(device)
            edge_sequence = edge_sequence.to(device)
            batch_select = batch_select.to(device)
            
            optimizer.zero_grad()
            
            logits = our_model(stock_sequence, edge_sequence,batch_select) 
            loss = loss_fn(logits, ret_mean_batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print(stocks_embedding.size())
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.detach().cpu().numpy())
            rets.append(ret_mean_batch.detach().cpu().numpy())
        print("Ep:",ep,"Loss:",total_loss)
        preds = np.concatenate(preds)
        rets = np.concatenate(rets)
        print('in Sample')
        scores = {
                  'weightied F1':f1_score(rets,preds, average = 'weighted'),
                  'macro F1': f1_score(rets,preds, average = 'macro'),
                  'micro F1': f1_score(rets,preds, average = 'micro')
                  }
        print(scores)
        print([sum(preds==i) for i in range(3)],[sum(rets==i) for i in range(3)])
        
        
        with torch.no_grad():
            preds_ = []
            rets_ = []
            for stock_sequence, ret_mean_batch, edge_sequence, select in  test_dl:
                stock_sequence = stock_sequence.to(device)
                ret_mean_batch = ret_mean_batch.to(device)
                edge_sequence = edge_sequence.to(device)
                select = select.to(device)
                preds = []
                rets = []
                logits = our_model(stock_sequence, edge_sequence,select) 
                pred = torch.argmax(logits, dim=1)
                preds_.append(pred.cpu())
                rets_.append(ret_mean_batch.cpu())
            preds_ = np.concatenate(preds_)
            rets_ = np.concatenate(rets_)
            print('out Sample')
            scores_ = {
                    'weightied F1':f1_score(rets_,preds_, average = 'weighted'),
                    'macro F1': f1_score(rets_,preds_, average = 'macro'),
                    'micro F1': f1_score(rets_,preds_, average = 'micro')
                    }
            print(scores_)
        print([sum(preds_==i) for i in range(3)],[sum(rets_==i) for i in range(3)])
        
