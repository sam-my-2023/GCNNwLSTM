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

from torch_geometric.utils import spmm,dense_to_sparse,to_dense_adj

# optuna tunning
import optuna
from optuna.trial import TrialState

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

def adjacent_matrix(positive_rel,negative_rel,TICKERS_IND, relation_types = ['positive']):
    m = len(TICKERS_IND)
    tA = np.zeros([m,m])
    r = len(positive_rel)
    assert r==len(negative_rel)

 # positive
    if 'positive' in relation_types:
        for graph in positive_rel:
            A = np.zeros([m,m])
            
            for k in range(len(graph)):
                i = TICKERS_IND[graph[k]]
                for c in graph[k+1:]:
                    j = TICKERS_IND[c]
                    tA[i,j] = 1
                    tA[j,i] = 1

 # negative set

    if 'negative' in relation_types:
        for graph in negative_rel:
            A = np.zeros([m,m])
            
            for k in range(len(graph)):
                i = TICKERS_IND[graph[k]]
                for c in graph[k+1:]:
                    j = TICKERS_IND[c]
                    tA[i,j] = 1
                    tA[j,i] = 1

    return tA

class Our_Dataset:
    def __init__(self,data_per_date_per_company,tag_per_date_per_company, adj_per_date, l, mask_size, recurrent_training = True):
        
        self.n = min(len(adj_per_date),                    
                     data_per_date_per_company.shape[0],
                     tag_per_date_per_company.shape[0])
        
        self.num_companies = data_per_date_per_company.shape[1]
        assert self.num_companies == tag_per_date_per_company.shape[1]
        self.adj_per_date = adj_per_date
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
        return self.data_per_date_per_company[j:j+self.l,:], tag, self.adj_per_date[j:j+self.l], select
    
    def __len__(self):
        return (self.n-self.l)*self.mask_number
    

def our_collate_fn(batch):
    #  bacth_size, seq_len,num_stocks,feature_size 
    inps = torch.tensor(np.stack([x[0] for x in batch],axis=0),dtype=torch.float32)

    corr = torch.tensor(np.concatenate([x[1] for x in batch],axis=1), dtype=torch.long)
    corr = corr.flatten()
    

    batch_n = inps.size(0)
    seq_m = inps.size(1)
    num_stocks = inps.size(2)
    
    # bacth_size,seq_len, num_stocks,num_stocks
    batch_seq_adj =torch.tensor(np.stack([x[2] for x in batch],axis=0), dtype=torch.long) 
    
    batch_select = torch.tensor(np.concatenate([ i*num_stocks+x[3] for i,x in enumerate(batch)]),dtype=torch.long)
    
    return inps, corr, batch_seq_adj, batch_select

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
        
        self.gcnn = GCNConv(in_channel, graph_channel)
        
        self.graph_embedding_flag = False
        
        if lstm_out_channel[0]:
            self.graph_embedding_flag = True
            self.comb_lstm = nn.LSTM(graph_channel + in_channel, lstm_out_channel[0], batch_first = True,num_layers = num_lstm_layer[0])
            combo_lstm_out = num_lstm_layer[0]*lstm_out_channel[0]
        else:
            combo_lstm_out = 0
        
        
        self.stock_lstm = nn.LSTM(in_channel, lstm_out_channel[1], batch_first = True, num_layers = num_lstm_layer[1])
        
        
        self.recurrent_training = recurrent_training
        if not recurrent_training:
            stock_lstm_out = num_lstm_layer[1]*lstm_out_channel[1]
        else:
            stock_lstm_out = lstm_out_channel[1]
            
        lstm_out = stock_lstm_out+combo_lstm_out
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
        
    def forward(self, inputs, batch_seq_adj, select):
        '''
        inputs: (batch_size, seq_length, num_stocks, feature_size)
    
        '''
        batch_seq_stock_size = inputs.size()[:3]
        inputs = self.input_mlp(inputs)
        
        stock_input = inputs.permute((0,2,1,3))
        stock_input = stock_input.flatten(end_dim=1)[select]
        stock_o, (stock_h,stock_c) = self.stock_lstm(stock_input)
        
        if self.recurrent_training:
            output = stock_o
        else:
            output = stock_h.permute((1,0,2))
            output = output.flatten(start_dim=1)
        
        if self.graph_embedding_flag:
            inputs = inputs.flatten(end_dim=2)
            batch_seq_adj = batch_seq_adj.flatten(end_dim=1)
        
            # sum of (sum edges in seq adj) over batch
            edge_index,_ = dense_to_sparse(batch_seq_adj)
            
            # batch_size*seq_len*num_stocks,garph_hidden_feature_size
            graph_embedding = self.gcnn(inputs,edge_index) 

            graph_embedding = torch.cat((graph_embedding,inputs),dim=1)
            graph_embedding = graph_embedding.unflatten(dim=0, sizes=batch_seq_stock_size )
            graph_embedding = graph_embedding.permute((0,2,1,3))
            graph_embedding = graph_embedding.flatten(end_dim=1)[select]
    
            comb_o, (comb_h,comb_c) = self.comb_lstm(graph_embedding) 
            
        
            if self.recurrent_training:
                output = torch.cat((comb_o,output),dim=-1)
            else:
                tem = comb_h.permute((1,0,2))
                output = torch.cat((tem.flatten(start_dim=1),output),dim=1)
        
        output = self.output_MLP(output)
        
        if self.recurrent_training:
            output = output.permute(1,0,2)
            output = output.flatten(end_dim=1)
        return F.log_softmax(output,dim=1)


def train():
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
    
    r = 20
    n = len(positive_rel)
    
    tA = np.stack([adjacent_matrix(positive_rel[i-r:i], negative_rel[i-r:i],TICKERS_IDX) for i in range(r,n+1)])

    n = min(price.shape[0],stock_feature.shape[0], n)
    m = len(TICKERS)
    
    
    # test part 
    
    positive_rel = test_json_wrapper.get_graphs()
    negative_rel = test_json_wrapper.get_graphs(relation_type='negative')
    
    n_test = len(positive_rel)
    
    test_tA = np.stack([adjacent_matrix(positive_rel[i-r:i], negative_rel[i-r:i],TICKERS_IDX) for i in range(r,n_test+1)])
    
    sep_point = n
    n_test = min(price.shape[0]-sep_point, stock_feature.shape[0]-n_test,n_test)
    
    end_point = n_test+sep_point
    
    l = 60
    batch_size = 16
    mask_size = 5
    recurent = False
    

    our_data_set = Our_Dataset(stock_feature[r-1:sep_point], tag[r-1:sep_point], tA[:sep_point+1-r], l,mask_size, recurrent_training=recurent)
    dataloader = DataLoader(our_data_set, batch_size, collate_fn = our_collate_fn, shuffle= True)
    

    test_ds = Our_Dataset(stock_feature[sep_point+r-1:end_point], tag[sep_point+r-1:end_point], test_tA[:n_test+1-r], l, mask_size,recurrent_training=recurent)
    test_dl = DataLoader(test_ds, batch_size, collate_fn = our_collate_fn, shuffle= False)
    
    device = 'cuda'
    in_MLP = None #[[128,None],[128,None]]
    out_MLP = [[512,None],[128,None],[128,None]]
    our_model = Our_Model(22,256,[128,256],3,[4,4],in_MLP ,out_MLP,recurrent_training=recurent).to(device)
    optimizer = optim.RMSprop(our_model.parameters(), lr = 1e-4, momentum= 0.9)
    # optimizer = optim.Adam(our_model.parameters(), lr=1e-6,momentum=0.9)
    # optimizer = optim.SGD(our_model.parameters(), lr = 1e-7,momentum=0.1)
    # optimizer = optim.Adadelta(our_model.parameters())
    
    loss_fn = nn.NLLLoss()
 

    for ep in range(10):
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
        
if __name__=="__main__":
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
    
    r = 20
    n = len(positive_rel)
    
    tA = np.stack([adjacent_matrix(positive_rel[i-r:i], negative_rel[i-r:i],TICKERS_IDX) for i in range(r,n+1)])

    n = min(price.shape[0],stock_feature.shape[0], n)
    m = len(TICKERS)
    
    
    # test part 
    
    positive_rel = test_json_wrapper.get_graphs()
    negative_rel = test_json_wrapper.get_graphs(relation_type='negative')
    
    n_test = len(positive_rel)
    
    test_tA = np.stack([adjacent_matrix(positive_rel[i-r:i], negative_rel[i-r:i],TICKERS_IDX) for i in range(r,n_test+1)])
    
    sep_point = n
    n_test = min(price.shape[0]-sep_point, stock_feature.shape[0]-n_test,n_test)
    
    end_point = n_test+sep_point
    
    l = 15
    batch_size = 16
    mask_size = 5
    recurent = False
    

    our_data_set = Our_Dataset(stock_feature[r-1:sep_point], tag[r-1:sep_point], tA[:sep_point+1-r], l,mask_size, recurrent_training=recurent)
    dataloader = DataLoader(our_data_set, batch_size, collate_fn = our_collate_fn, shuffle= True)
    

    test_ds = Our_Dataset(stock_feature[sep_point+r-1:end_point], tag[sep_point+r-1:end_point], test_tA[:n_test+1-r], l, mask_size,recurrent_training=recurent)
    test_dl = DataLoader(test_ds, batch_size, collate_fn = our_collate_fn, shuffle= False)
    
    def objective(trial):
        # graph_hidden = trial.suggest_categorical("graph_hidden", [2 ** i for i in range(6,10)])
        # forward_aggr = trial.suggest_categorical("forward_aggr", ['max','mean','add'])
        # back_aggr = trial.suggest_categorical("back_aggr", ['max','mean','add'])
        learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        mt = trial.suggest_float("momentum", 0.1, 0.9)
        
        device = 'cuda'
        in_MLP = None #[[128,None],[128,None]]
        out_MLP = [[512,None],[128,None],[128,None]]
        our_model = Our_Model(22,256,[128,256],3,[4,4],in_MLP ,out_MLP,recurrent_training=recurent).to(device)
        optimizer = optim.RMSprop(our_model.parameters(), lr = learning_rate, momentum= mt)
        # optimizer = optim.Adam(our_model.parameters(), lr=1e-6,momentum=0.9)
        # optimizer = optim.SGD(our_model.parameters(), lr = 1e-7,momentum=0.1)
        # optimizer = optim.Adadelta(our_model.parameters())
        
        loss_fn = nn.NLLLoss()
    

        for ep in range(10):
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
            preds = np.concatenate(preds)
            rets = np.concatenate(rets)
            scores = {
                    'weightied F1':f1_score(rets,preds, average = 'weighted'),
                    'macro F1': f1_score(rets,preds, average = 'macro'),
                    'micro F1': f1_score(rets,preds, average = 'micro')
                    }
            
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
                scores_ = {
                        'weightied F1':f1_score(rets_,preds_, average = 'weighted'),
                        'macro F1': f1_score(rets_,preds_, average = 'macro'),
                        'micro F1': f1_score(rets_,preds_, average = 'micro')
                        }
        
            trial.report(scores_['macro F1'], ep)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            global nn_parameters
            nn_parameters = our_model
        return scores['macro F1']
    
    def callback(study, trial):
        global best_nn_parameters
        if study.best_trial == trial:
            best_nn_parameters = nn_parameters
    
    study_name = "gcnn_lstm_"+'seq_'+str(l)+'_r_'+str(r)  # Unique identifier of the study.
    storage_name = "postgresql+psycopg2://optuna_study:tiger@localhost:5432/study_storage"
    study = optuna.create_study(direction="maximize",study_name=study_name, storage=storage_name,load_if_exists=True)
    study.optimize(objective, n_trials=256 , n_jobs=4,callbacks=[callback])
    torch.save(best_nn_parameters.state_dict(), 'nn_weights'+study_name )