import pandas as pd
import numpy as np
import os
import torch
import sys, os
from .data_process import stock_time_series 
from torch.utils.data import DataLoader
from .data_process import graph_time_series
from .data_process.graph_builder import graph_builder
from .nn_model import graph_nn
from torch_geometric.utils import negative_sampling
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
num_stocks = 30
num_news = 40

class Our_Dataset:
    def __init__(self,data_per_date_per_company,tag_per_date_per_company, egde_per_date, l):
        num_companies = data_per_date_per_company.shape[1]
        assert num_companies == tag_per_date_per_company.shape[1]
        self.egde_per_date = egde_per_date
        self.data_per_date_per_company = data_per_date_per_company
        self.tag_per_date_per_company = tag_per_date_per_company
        self.l = l

        self.n = min(len(egde_per_date),                    
                     data_per_date_per_company.shape[0],
                     tag_per_date_per_company.shape[0])
    
    def __getitem__(self,i):
        assert i<(self.n-self.l)
        return self.data_per_date_per_company[i:i+self.l,:], self.tag_per_date_per_company[i+self.l], self.egde_per_date[i:i+self.l]
    
    def __len__(self):
        return self.n-self.l
    

def our_collate_fn(batch):
    inps = torch.tensor(np.concatenate([x[0] for x in batch],axis=1),dtype=torch.float32, device = DEVICE)
    corr = torch.tensor(np.stack([x[1] for x in batch],axis=0), dtype=torch.float32, device = DEVICE)
    tem = []

    for i in range(len(batch)):
        # news_clusters = 5, num_companies = 30
        tem.append([[i*num_news+x[0],i*num_stocks+x[1]] for x in batch[i][2]])
    edges = torch.tensor(np.concatenate(tem,axis=2), dtype=torch.long, device = DEVICE)
    return inps, corr, edges 



def train(kwargs,l,k_log,kh_cov,batch_size = 4,lr=1e-5,ret_low=1.0,prob = 0.5):
    TICKERS = ['HD', 'INTC', 'JPM', 'CAT', 'WMT', 'HON', 'JNJ', 'AXP', 'DIS', 'UNH', 'CVX', 'PG', 'NKE', 'IBM', 'CRM', 'AMGN', 'KO', 'BA', 'CSCO', 'TRV', 'AAPL', 'MSFT', 'DOW', 'MRK', 'MMM', 'WBA', 'V', 'GS', 'MCD', 'VZ']
    price = stock_time_series.get_stock_price(TICKERS)
    price = price.squeeze()
    n, m = price.shape
    # ret = price[k_log:]/price[:n-k_log]
    # log_ret = np.log(price[k_log:]/price[:n-k_log])
    # n = n-k_log

    # correlation = np.array( [np.corrcoef(log_ret[i-kh_cov:i+kh_cov],rowvar=False) for i in range(kh_cov,n-kh_cov)] )
    
    ret_mean = np.array([(np.mean(price[i-kh_cov:i+kh_cov]/price[i-k_log]>=ret_low,axis=0)>=prob ).astype(int) for i in range(k_log,n-kh_cov)] )
    n=n-kh_cov-k_log
    
    stock_feature = stock_time_series.get_stock_feature(TICKERS)[:n]
    our_formats = graph_time_series.news_ticker_our_formats(TICKERS,graph_builder)[:n]
    
    # _,stocks_num,feature_dim = stock_feature.shape
    
    train_n = len(our_formats)
    print(train_n)
 
    our_data_set = Our_Dataset(stock_feature,ret_mean , our_formats, l)
    dataloader = DataLoader(our_data_set, batch_size, collate_fn = our_collate_fn, shuffle= True)
    kwargs['args_2'] = [num_stocks,num_news]
    our_model = graph_nn.Our_Model(**kwargs).to(DEVICE)
    # loss_fn = nn.MSELoss()
    # loss_fn = F.nll_loss
    loss_fn = nn.BCELoss()
    
    optimizer = optim.RMSprop(our_model.parameters(), lr)
    for ep in range(10):
        preds = []
        rets = []
        for stock_sequence, ret_mean_batch, edge_sequence in dataloader:
            optimizer.zero_grad()
            # pred = our_model(stock_sequence,edge_sequence)
            logits = our_model(stock_sequence, edge_sequence)
            # loss = corr_sample_loss(sample_edges,stocks_embedding)
            # loss = loss_fn(pred[-1],ret_mean)
            ret_mean_batch  = ret_mean_batch.flatten()
            
            loss = loss_fn(logits, ret_mean_batch)
            loss.backward()
            optimizer.step()
            # print(stocks_embedding.size())
            pred = logits>0.5
            preds.append(pred.cpu().detach().numpy())
            rets.append(ret_mean_batch.cpu().detach().numpy())
        print(loss)
        preds = np.concatenate(preds)
        rets = np.concatenate(rets)
        print(np.mean(preds==rets))
    return preds,rets,ret_mean

if __name__=="__main__":
    train()
    


    
