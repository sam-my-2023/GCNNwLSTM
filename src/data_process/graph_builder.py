import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def graph_builder(news_embedding,company_embedding):
    news_to_news_siminarity = cosine_similarity(news_embedding,news_embedding)
    num_news = 40
    news_id = [0]
    k = 1
    while len(news_id)<num_news and len(news_id)>=k:
        for i in np.argsort(news_to_news_siminarity[news_id[-k]]):
            if max([news_to_news_siminarity[i][j] for j in news_id])>0.85:
                continue
            else:
                k = 1
                news_id.append(i)
                break
    news_embedding = np.array([news_embedding[i] for i in news_id])
    news_stock_siminarity = cosine_similarity(news_embedding,company_embedding)
    k = 10
    rst = np.concatenate([[i*np.ones(k), np.argsort(x)[-1:-k-1:-1]] for i,x in enumerate(news_stock_siminarity)],axis=1)
    return rst #np.transpose(rst,(0,2,1))

