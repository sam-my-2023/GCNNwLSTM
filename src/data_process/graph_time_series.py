import pickle
import os
import sys
import json
import numpy as np
from scipy.sparse.linalg import svds
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDING_PATH  = os.path.abspath('/home/sam/mingsong/ChatgptGraph/chatgpt_embedding')
EMBEDDING_MODEL = "text-embedding-ada-002"

path_to_embedding_folder = os.path.join(EMBEDDING_PATH,EMBEDDING_MODEL)
path_to_company_embedding =os.path.join(EMBEDDING_PATH,'company_info_embedding.pkl')

# def get_company_index_in_embedding(tickers, path_to_company_info):
#     with open(path_to_company_info,'r') as f:
#         nasdaq_constituent = json.load(f)
#     tem_dict = { c['symbol']:i for i,c in enumerate(nasdaq_constituent)}
#     return [tem_dict[s] for s in tickers]

def newstockgraph_svd(eigenvector):
    n,m = eigenvector.shape
    # n == k of svd
    # m == num of companies
    rst = []
    for component in eigenvector:
        mask = [0]*m
        for i in np.argsort(component)[-1:-6:-1]:
            mask[i] = 1
        rst.append(np.array(mask))
    return np.array(rst)

def our_formats_from_siminarity(hl_matrixs,company_matrix,graph_builder = None):
    print("transmiting the siminarity to our formats")
    rst = []
    if graph_builder is None:
        for h in tqdm(hl_matrixs):
            siminarity_matrix = cosine_similarity(h,company_matrix)
            # A=u@s@vT, vT combines the stock by portion
            # A v_i = sigma_i u_i
            u,s,vT = svds(siminarity_matrix, k = 5)
            
            rst.append(newstockgraph_svd(vT))
        

def news_ticker_our_formats(tickers, graph_builder,
                            path_news_folder=path_to_embedding_folder,
                            path_company_embedding = path_to_company_embedding,
                            ):
    
    with open(path_company_embedding, 'rb') as f:
        company_info_emedding=pickle.load(f)
    company_matrix = np.array(company_info_emedding)
    company_matrix = np.squeeze(company_matrix)
    
    print("loading embedding matrixs")
    hl_matrixs = []
    num_raw_news = len(os.listdir(path_news_folder))
    for i in range(num_raw_news-1):
        tem_path = os.path.join(path_news_folder,'day_'+str(i)+'.pkl')
        with open(tem_path, 'rb') as f:
            hl_matrix = pickle.load(f)
            hl_matrix = np.squeeze(hl_matrix)
            hl_matrixs.append(hl_matrix)
    print("generating the graph in our formats")
    rst = []
    for h in tqdm(hl_matrixs):
        if h.shape[0]<1000:
            tem = graph_builder(h,company_info_emedding)
        rst.append(tem)
    print("Finished!")
    return rst