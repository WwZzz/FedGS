import numpy as np
from .fedbase import BasicServer
from .fedbase import BasicClient as Client
import collections
import copy
import networkx as nx
import utils.systemic_simulator as ss
import utils.fflow as flw
import gurobipy as grb
from utils import fmodule
import torch

def softmax(x, t=1):
    x = np.array(x)
    ex = np.exp(x/t)
    sum_ex = np.sum(ex)
    return ex / sum_ex

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.algo_para = {'alpha':0.1}
        self.init_algo_para(option['algo_para'])
        self.ftr = 'update'
        self.epsilon = 0.1
        self.sigma = 0.01
        # client sampling frequency
        self.sample_frequency = [0 for _ in range(self.num_clients)]

    def iterate(self, t):
        if self.current_round==0:
            # initialize 3DG by assuming all the clients are available in the beginning
            self.models = self.communicate([cid for cid in range(self.num_clients)])['model']
            cossim_features = self.get_client_feature(self.ftr)
            # create oracle graph from client features with similarity function `cossim`
            self.graph = self.create_graph_from_features(cossim_features, fsim=lambda x, y: np.dot(x, y) / np.sqrt(((x ** 2).sum()) * ((y ** 2).sum())))
            self.adj = self.create_adj_from_graph(self.graph)
            # init shortest dist matrix
            self.shortest_dist, _ = self.floyd(self.adj)
        else:
            # init node
            self.selected_clients = self.sample()
            # training
            models = self.communicate(self.selected_clients)['model']
            # aggregate: pk = 1/K as default where K=len(selected_clients)
            self.model = self.aggregate(models)
            for cid in self.selected_clients: self.sample_frequency[cid]+=1
        return

    @ss.with_inactivity
    def sample(self):
        if len(self.active_clients) == 1: return self.active_clients
        N = len(self.active_clients)
        M = min(len(self.active_clients), self.clients_per_round)
        Ht = 2.0/(M*(M-1))*self.shortest_dist[self.active_clients,:][:,self.active_clients]
        Ht = self.alpha * Ht
        # filling up H's diag
        vt = np.array(self.sample_frequency)
        v_mean = vt.mean()
        ut = 2*(vt-v_mean-1.0*M/N)+1
        ut = -1.0/(self.num_clients-1)*ut[self.active_clients]
        for i in range(N):
            Ht[i][i] = ut[i]
        # optimize
        m = grb.Model(name="MIP Model")
        used = [m.addVar(vtype=grb.GRB.BINARY) for _ in range(N)]
        objective = grb.quicksum(Ht[i, j] * used[i] * used[j] for i in range(0, N) for j in range(i, N))
        m.addConstr(
            lhs=grb.quicksum(used),
            sense=grb.GRB.EQUAL,
            rhs=M
        )
        m.ModelSense = grb.GRB.MAXIMIZE
        m.setObjective(objective)
        m.setParam('OutputFlag', 0)
        m.Params.TimeLimit = 5
        m.optimize()
        # convert result
        res = [xi.X for xi in used]
        selected_clients = []
        for cid,flag in zip(self.active_clients,res):
            for _ in range(int(flag)):
                selected_clients.append(cid)
        return selected_clients

    def aggregate(self, models):
        datavols = np.array(self.local_data_vols)[self.selected_clients]
        p = datavols / self.total_data_vol
        sump = np.sum(p)
        p = p/sump
        return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])

    def get_client_feature(self, key='label'):
        if key=='oracle':
            if self.option['task'].startswith('synthetic_regression'):
                client_features = []
                for c in self.clients:
                    client_features.append(np.array(c.train_data.optimal_model).reshape(-1))
            else:
                client_features = []
                for c in self.clients:
                    labels = collections.Counter([c.train_data[did][-1] for did in range(len(c.train_data))])
                    # cal labels distribution
                    dict = {}
                    for i in range(10):
                        if i not in labels.keys():
                            dict[i] = 0
                        else:
                            dict[i] = labels[i]
                    labels = dict
                    lb_dist = [i / len(c.train_data) for i in labels.values()]
                    client_features.append(lb_dist)
        elif key=='label':
            client_features = []
            for c in self.clients:
                labels = collections.Counter([c.train_data[did][-1] for did in range(len(c.train_data))])
                # cal labels distribution
                dict = {}
                for i in range(10):
                    if i not in labels.keys():
                        dict[i] = 0
                    else:
                        dict[i] = labels[i]
                labels = dict
                lb_dist = [i / len(c.train_data) for i in labels.values()]
                client_features.append(lb_dist)
        elif key=='update':
            models = [m-self.model for m in self.models]
            client_features = [fmodule._model_to_tensor(m).cpu().numpy() for m in models]
        elif key=='logit':
            data_loader = iter(self.calculator.get_data_loader(self.test_data, batch_size=128))
            batch_data = next(data_loader)
            noise = torch.normal(mean=torch.mean(batch_data[0]), std=torch.std(batch_data[0]), size=batch_data[0].shape).to(self.device)
            client_features = []
            for model in self.models:
                random_output = model(noise)
                mean_out = random_output.mean(axis=0)
                client_features.append(mean_out.cpu().detach().numpy())
        elif key=='embedding':
            data_loader = iter(self.calculator.get_data_loader(self.test_data, batch_size=self.b))
            batch_data = next(data_loader)
            noise = torch.normal(mean=torch.mean(batch_data[0]), std=torch.std(batch_data[0]), size=batch_data[0].shape).to(self.device)
            client_features = []
            for model in self.models:
                random_output = model.get_embedding(noise).view(len(random_output), -1)
                mean_out = random_output.mean(axis=0)
                client_features.append(mean_out.cpu().detach().numpy())
        return client_features

    def create_adj_from_graph(self, graph, epsilon=0.01, sigma=0.1):
        n = len(graph)
        adj = np.zeros((n, n))
        for ci in range(n):
            adj[ci][ci] = 0
            for cj in range(ci + 1, n):
                if graph[ci][cj] > epsilon:
                    adj[ci][cj] = adj[cj][ci] = np.exp(-graph[ci][cj] ** 2 / sigma)
                else:
                    adj[ci][cj] = adj[cj][ci] = np.inf
        return adj

    def create_graph_from_features(self, features, fsim=None):
        if fsim==None: fsim=np.dot
        n = len(features)
        oracle_graph = np.zeros((n,n))
        for ci in range(n):
            for cj in range(ci + 1, n):
                sim = fsim(np.array(features[ci]), np.array(features[cj]))
                oracle_graph[ci][cj] = oracle_graph[cj][ci] = sim
        # normalize the similarity
        max_sim = np.max(oracle_graph)
        min_sim = np.min(oracle_graph)
        for ci in range(n):
            oracle_graph[ci][ci] = max_sim
        oracle_graph = (oracle_graph - min_sim)/(max_sim - min_sim)
        for ci in range(n):
            oracle_graph[ci][ci] = np.inf
        return oracle_graph

    def floyd(self, adj):
        n = len(adj)
        path = np.zeros((n, n))
        dis = copy.deepcopy(adj)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dis[i][k] + dis[k][j] < dis[i][j]:
                        dis[i][j] = dis[i][k] + dis[j][k]
                        path[i][j] = k
        # normalize shortest distance matrix as
        cp_dis = dis.copy()
        cp_dis[np.isinf(dis)] = -1
        max_available_dist = np.max(cp_dis)
        dis = dis/max_available_dist
        maxv = np.max(dis[np.where(dis!=np.inf)])
        for i in range(self.num_clients):
            for j in range(i, self.num_clients):
                if np.isinf(dis[i][j]):
                    dis[i][j] = dis[j][i] = maxv
        return dis, path
