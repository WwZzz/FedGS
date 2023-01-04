import matplotlib
import numpy as np
from .fedbase import BasicServer
from .fedavg import Client
import matplotlib.pyplot as plt
import collections
import copy
import networkx as nx
import utils.systemic_simulator as ss
from utils import fmodule
import torch

def softmax(x, t=1):
    x = np.array(x)
    ex = np.exp(x/t)
    sum_ex = np.sum(ex)
    return ex / sum_ex

def cossim(x, y):
    cossim = np.dot(x, y) / np.sqrt(((x ** 2).sum()) * ((y ** 2).sum()))
    return max(cossim, 0)

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.nodes = [str(cid) for cid in range(self.num_clients)]
        self.colors = ['r', 'b', 'y', 'g', 'black', 'pink', 'orange', 'gold', 'lightskyblue', '#ffb3e6']

    def iterate(self, t):
        self.selected_clients = [cid for cid in range(self.num_clients)]
        models = self.communicate(self.selected_clients)['model']
        self.models = models
        # compute features
        oracle_features = self.get_client_feature('label')
        cossim_features = self.get_client_feature('update')
        functional_features = self.get_client_feature('logit')
        # compute similarity matrix
        oracle_sim = self.create_similarity_matrix_from_features(oracle_features)
        cossim_sim = self.create_similarity_matrix_from_features(cossim_features, cossim)
        functional_sim = self.create_similarity_matrix_from_features(functional_features, cossim)
        # build up 3DG
        epsilon = 0.1
        sigma2 = 0.01
        oracle_3DG = self.create_graph_from_similarity(oracle_sim, epsilon=epsilon, sigma=sigma2)
        es = [0, 0.01, 0.05, 0.1, 0.2, 0.5]
        res_func = []
        res_cos = []
        for ep in es:
            cossim_3DG = self.create_graph_from_similarity(cossim_sim, epsilon=ep, sigma=sigma2)
            func_3DG = self.create_graph_from_similarity(functional_sim, epsilon=ep, sigma=sigma2)
        # measurement
            func_metric = self.measure_3DG_construction(oracle_3DG, func_3DG)
            cos_metric = self.measure_3DG_construction(oracle_3DG, cossim_3DG)
            res_func.append(func_metric)
            res_cos.append(cos_metric)
        print(res_func)
        print(res_cos)
        return

    def get_client_feature(self, key='label'):
        if key=='label':
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

    def create_graph_from_similarity(self, graph, epsilon=0.01, sigma=0.1):
        n = len(graph)
        adj = np.zeros((n, n))
        for ci in range(n):
            for cj in range(ci + 1, n):
                if graph[ci][cj] > epsilon:
                    adj[ci][cj] = adj[cj][ci] = np.exp(-graph[ci][cj] ** 2 / sigma)
                else:
                    adj[ci][cj] = adj[cj][ci] = np.inf
        return adj

    def create_similarity_matrix_from_features(self, features, fsim=None):
        if fsim==None: fsim=np.dot
        n = len(features)
        oracle_graph = np.zeros((n,n))
        for ci in range(n):
            oracle_graph[ci][ci] = 1
            for cj in range(ci + 1, n):
                sim = fsim(np.array(features[ci]), np.array(features[cj]))
                oracle_graph[ci][cj] = oracle_graph[cj][ci] = sim
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
        return dis, path

    def measure_3DG_construction(self, oracle_adj, adj):
        n = len(adj)
        obj_edges = [(i, j) for i in range(n) for j in range(i, n) if
                     oracle_adj[i][j] >= 0 and oracle_adj[i][j] != np.inf]
        pred_edges = [(i, j) for i in range(n) for j in range(i, n) if adj[i][j] >= 0 and adj[i][j] != np.inf]
        intersection = set(pred_edges).intersection(obj_edges)
        precision = len(intersection) / len(pred_edges)
        recall = len(intersection) / len(obj_edges)
        return precision, recall, 2.0/(1.0/precision+1.0/recall)

    def gen_nxgraph_from_adj(self, adj):
        n = len(adj)
        nodes = [str(i) for i in range(n)]
        edges = [(str(ci), str(cj)) for ci in range(n) for cj in range(ci + 1, n) if adj[ci][cj] != np.inf]
        G = nx.Graph()
        for node in nodes:
            G.add_node(node)
        G.add_edges_from(edges)
        return G

    def draw_graph_with_pie(self, G, pie_node=None, pie_size=0.1, node_labels=None, colors=None, label_delta=[0.0, 0.0], with_selected=True):
        if colors==None: colors = [c for c in matplotlib.colors.CSS4_COLORS.keys()]
        fig = plt.figure(figsize=(8,8))
        # get the position of nodes
        pos = nx.spring_layout(G)
        # draw edges and nodes
        nx.draw_networkx_edges(G, pos=pos)
        # draw node labels
        if node_labels is not None:
            if with_selected:
                labels = {node:str(node_labels[id]) for id, node in enumerate(G.nodes)}
                for node, nodelb in labels.items():
                    nid = int(node)
                    if nid in self.selected_clients:
                        labels[node] = str(int(nodelb)-1) + '+1'
            else:
                labels = {node:str(node_labels[id]) for id, node in enumerate(G.nodes)}
            label_pos = {k: v + label_delta for k, v in pos.items()}
            nx.draw_networkx_labels(G, label_pos, labels=labels, font_size=12)
        # draw pie instead of node
        if pie_node is not None:
            colors = colors[:len(pie_node[0])]
            for node_name, node_pos in pos.items():
                nid = int(node_name)
                if nid==0:
                    plt.pie(pie_node[nid], center=node_pos, colors=colors, radius=pie_size, labels=[str(i) for i in range(len(colors))], textprops={'color':'none'})
                else:
                    plt.pie(pie_node[nid], center=node_pos, colors=colors, radius=pie_size)
            plt.legend()
        if with_selected:
            for node_name, node_pos in pos.items():
                nid = int(node_name)
                if nid in self.selected_clients:
                    # s > pi * pie_size^2 / 4
                    s = 4 * 3.1415 * (pie_size**2) /4 * 400
                    plt.plot(node_pos[0], node_pos[1], marker='o', markeredgecolor='r', linestyle='none', markerfacecolor='none', markersize=20)
                    # plt.scatter([node_pos[0]], [node_pos[1]], marker='o',s=s, c='none', edgecolor='black', linewidth=10)
        # adjust the graph to the center
        px, py = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
        cx, cy = (np.min(px)+np.max(px))/2, (np.min(py)+np.max(py))/2
        plt.axis('off')
        axis = plt.gca()
        crt_xlim, crt_ylim = axis.get_xlim(), axis.get_ylim()
        dx, dy = cx - (crt_xlim[0]+crt_xlim[1])/2, cy - (crt_ylim[0]+crt_ylim[1])/2
        axis.set_xlim([1.2 * x for x in (crt_xlim[0]+dx, crt_xlim[1]+dx)])
        axis.set_ylim([1.2 * y for y in (crt_ylim[0]+dy, crt_ylim[1]+dy)])
        plt.tight_layout()
        plt.show()
        return
