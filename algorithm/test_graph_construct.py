import matplotlib
import numpy as np
from .fedbase import BasicServer
from .fedavg import Client
import matplotlib.pyplot as plt
import collections
import copy
import networkx as nx
import utils.systemic_simulator as ss
import torch
import utils.fflow as flw

def softmax(x, t=1):
    x = np.array(x)
    ex = np.exp(x/t)
    sum_ex = np.sum(ex)
    return ex / sum_ex

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.algo_para = {'b': 64}
        self.init_algo_para(option['algo_para'])

    def iterate(self, t):
        self.selected_clients = [_ for _ in range(self.num_clients)]
        # training
        models = self.communicate(self.selected_clients)['model']
        # generate Gaussian Noise
        self.data_loader = iter(self.calculator.get_data_loader(self.test_data, batch_size=self.b))
        batch_data = next(self.data_loader)
        noise = torch.normal(mean=torch.mean(batch_data[0]), std=torch.std(batch_data[0]), size=batch_data[0].shape).to(self.device)
        self.client_features = []
        for model in models:
            random_output = model(noise)
            mean_out = random_output.mean(axis=0)
            self.client_features.append(mean_out.cpu().detach().numpy())
        self.graph = self.create_graph_from_features(self.client_features)
        self.adj = self.create_adj_from_graph(self.graph)
        self.nodes = [str(cid) for cid in range(len(self.clients))]
        self.edges = [(str(ci), str(cj)) for ci in range(len(self.clients)) for cj in range(ci + 1, len(self.clients))
                      if self.adj[ci][cj] != np.inf]
        self.colors = ['r', 'b', 'y', 'g', 'black', 'pink', 'orange', 'gold', 'lightskyblue', '#ffb3e6']
        self.sample_times = [0 for _ in self.clients]
        self.G = nx.Graph()
        for node in self.nodes:
            self.G.add_node(node)
        self.G.add_edges_from(self.edges)
        self.draw_graph_with_pie(flw.logger.client_features, colors=['r', 'b', 'y', 'g', 'black', 'pink', 'orange', 'gold', 'lightskyblue', '#ffb3e6'], pie_size=0.04, with_selected=False)
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models, p=[1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.selected_clients])
        return

    def create_graph_from_features(self, features, fsim=None):
        if fsim==None: fsim=np.dot
        n = len(features)
        oracle_graph = np.zeros((n,n))
        for ci in range(n):
            oracle_graph[ci][ci] = 1
            for cj in range(ci + 1, n):
                sim = fsim(np.array(features[ci]), np.array(features[cj]))
                oracle_graph[ci][cj] = oracle_graph[cj][ci] = sim
        return oracle_graph

    def create_adj_from_graph(self, graph, epsilon=0.01, sigma=0.1):
        n = len(graph)
        adj = np.zeros((n, n))
        for ci in range(n):
            for cj in range(ci + 1, n):
                if graph[ci][cj] > epsilon:
                    adj[ci][cj] = adj[cj][ci] = np.exp(-graph[ci][cj] ** 2 / sigma)
                else:
                    adj[ci][cj] = adj[cj][ci] = np.inf
        return adj

    def draw_graph_with_pie(self, pie_node=None, pie_size=0.1, node_labels=None, colors=None, label_delta=[0.0, 0.0], with_selected=True):
        if colors==None: colors = [c for c in matplotlib.colors.CSS4_COLORS.keys()]
        fig = plt.figure(figsize=(8,8))
        # get the position of nodes
        pos = nx.spring_layout(self.G)
        # draw edges and nodes
        nx.draw_networkx_edges(self.G, pos=pos)
        # draw node labels
        if node_labels is not None:
            if with_selected:
                labels = {node:str(node_labels[id]) for id, node in enumerate(self.G.nodes)}
                for node, nodelb in labels.items():
                    nid = int(node)
                    if nid in self.selected_clients:
                        labels[node] = str(int(nodelb)-1) + '+1'
            else:
                labels = {node:str(node_labels[id]) for id, node in enumerate(self.G.nodes)}
            label_pos = {k: v + label_delta for k, v in pos.items()}
            nx.draw_networkx_labels(self.G, label_pos, labels=labels, font_size=12)
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
        axis.set_xlim([x for x in (crt_xlim[0]+dx, crt_xlim[1]+dx)])
        axis.set_ylim([y for y in (crt_ylim[0]+dy, crt_ylim[1]+dy)])
        plt.tight_layout()
        plt.show()
        return