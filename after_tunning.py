from src.tunning_gcnn_lstm_nasdaq30 import *
PROJECT_PATH = os.path.abspath('..')
sys.path.append(PROJECT_PATH)

class Eval_Dataset:
    def __init__(self,data_per_date_per_company,tag_per_date_per_company, adj_per_date, l, mask_size, recurrent_training = True):
        
        self.n = min(adj_per_date.shape[0],                    
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
        if self.mask_size < self.num_companies:
            rng = np.random.default_rng()
            tem = np.vstack(( np.arange(self.num_companies), ) *self.n)
            self.random_select_per_day = rng.permuted(tem, axis=1)
        else:
            self.random_select_per_day = None
        self.recurrent_training = recurrent_training
    
    def __getitem__(self,i):
        j = i//self.mask_number
        k = i%self.mask_number
        if self.random_select_per_day:
            select = self.random_select_per_day[j,k*self.mask_size:(k+1)*self.mask_size]
            assert j<(self.n-self.l)
            if self.recurrent_training:
                tag = self.tag_per_date_per_company[j:j+self.l,select]
            else:
                tag = [self.tag_per_date_per_company[j+self.l,select]]
            return self.data_per_date_per_company[j:j+self.l,:], tag, self.adj_per_date[j:j+self.l], select
        else:
            return self.data_per_date_per_company[j:j+self.l,:], self.tag_per_date_per_company[j+self.l], self.adj_per_date[j:j+self.l]
    
    def __len__(self):
        return (self.n-self.l)*self.mask_number

class GraphEmbeddingOnly(nn.Module):
    def __init__(self, in_channel, graph_channel, lstm_out_channel, out_channel,num_lstm_layer,input_MLP=None, hidden_channels=None, recurrent_training = True):
        super().__init__()
        
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
    
    def forward(self, inputs, batch_seq_adj):
        '''
        inputs: (batch_size, seq_length, num_stocks, feature_size)
    
        '''
        batch_seq_stock_size = inputs.size()[:-1]
        inputs = self.input_mlp(inputs)
        
        # stock_input = inputs.permute((0,2,1,3))
        # stock_input = stock_input.flatten(end_dim=1)[select]
        # stock_o, (stock_h,stock_c) = self.stock_lstm(stock_input)
        
        # if self.recurrent_training:
        #     output = stock_o
        # else:
        #     output = stock_h.permute((1,0,2))
        #     output = output.flatten(start_dim=1)
        
        if self.graph_embedding_flag:
            inputs = inputs.flatten(end_dim=-2)
            batch_seq_adj = batch_seq_adj.flatten(end_dim=-2)
        
            # sum of (sum edges in seq adj) over batch
            edge_index,_ = dense_to_sparse(batch_seq_adj)
            
            # batch_size*seq_len*num_stocks,garph_hidden_feature_size
            gcnn_embedding = self.gcnn(inputs,edge_index)

            graph_embedding = torch.cat((gcnn_embedding,inputs),dim=1)
            graph_embedding = graph_embedding.unflatten(dim=0, sizes=batch_seq_stock_size )

            gcnn_embedding = gcnn_embedding.unflatten(dim=0, sizes=batch_seq_stock_size )

            
            if len(graph_embedding.size()) == 4:
                graph_embedding = graph_embedding.permute((0,2,1,3))
                graph_embedding = graph_embedding.flatten(end_dim=1)
                seq_graph_embedding = None
                seq_gcnn_embedding = None
            else:
                seq_gcnn_embedding = gcnn_embedding.permute((1,0,2))
                seq_graph_embedding = graph_embedding
                graph_embedding = graph_embedding.permute((1,0,2))
            assert(len(graph_embedding.size()) == 3)
    
            comb_o, (comb_h,comb_c) = self.comb_lstm(graph_embedding)
            tem = comb_h.permute((1,0,2))

            return seq_gcnn_embedding, seq_graph_embedding, tem.flatten(start_dim=1)
            
        
            # if self.recurrent_training:
            #     output = torch.cat((comb_o,output),dim=-1)
            # else:
            #     tem = comb_h.permute((1,0,2))
            #     output = torch.cat((tem.flatten(start_dim=1),output),dim=1)
        else:
            print('no graph embedding from init')
            return None

        # output = self.output_MLP(output)
        
        # if self.recurrent_training:
        #     output = output.permute(1,0,2)
        #     output = output.flatten(end_dim=1)
        # return F.log_softmax(output,dim=1)
                         


if __name__=="__main__":

    recurent = False

    PATH = '/home/sam/mingsong/ChatgptGraph/GCNNwLSTM'

    l = 60
    r = 1

    file_name = 'nn_weightsgcnn_lstm_seq_60'
    PATH = os.path.join(PATH, file_name)
    state_dict = torch.load(PATH)


    in_MLP = None #[[128,None],[128,None]]
    out_MLP = [[512,None],[128,None],[128,None]]
    our_model = GraphEmbeddingOnly(22,256,[128,256],3,[4,4],in_MLP ,out_MLP,recurrent_training=recurent)

    # load model
    our_model.load_state_dict(state_dict, strict=False)
    our_model.eval()

    # data

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
    price = torch.tensor(price, dtype=torch.float32)
    
    stock_feature = stock_time_series.get_stock_feature(TICKERS, os.path.join(PROJECT_PATH,'data'))
    stock_feature = torch.tensor(stock_feature, dtype=torch.float32)
    
    tag = np.stack([1*((price[i+1]/price[i])>=1.01) + 1*((price[i+1]/price[i])>.99) for i in range(price.shape[0]-1)])
    tag = torch.tensor(tag, dtype=torch.long)


    positive_rel = json_wrapper.get_graphs()
    negative_rel = json_wrapper.get_graphs(relation_type='negative')
    
    n = len(positive_rel)
    
    tA = np.stack([adjacent_matrix(positive_rel[i-r:i], negative_rel[i-r:i],TICKERS_IDX) for i in range(r,n+1)])
    tA = torch.tensor(tA, dtype= torch.long)

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
    
    
    batch_size = 1
    mask_size = 30
  
    

    our_data_set = Eval_Dataset(stock_feature[r-1:sep_point], tag[r-1:sep_point], tA[:sep_point+1-r], l,mask_size, recurrent_training=recurent)

    test_ds = Eval_Dataset(stock_feature[sep_point+r-1:end_point], tag[sep_point+r-1:end_point], test_tA[:n_test+1-r], l, mask_size,recurrent_training=recurent)
    
    gcnn_embedding, seq_stock,tag,seq_graph = our_data_set[0]
    
    seq_graph_embedding,lstm_hidden_embedding = our_model(seq_stock,seq_graph)

 