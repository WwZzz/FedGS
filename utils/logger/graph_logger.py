import utils.logger.basic_logger as bl
import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx

class Logger(bl.Logger):
    def initialize(self, *args, **kwargs):
        # get client's feature
        self.client_features = self.get_client_feature()
        # create oracle graph from client features
        self.oracle_graph = self.create_graph_from_features(self.client_features)
        self.adj = self.create_adj_from_graph(self.oracle_graph)
        self.nodes = [str(cid) for cid in range(len(self.clients))]
        self.edges = [(str(ci), str(cj)) for ci in range(len(self.clients)) for cj in range(ci+1, len(self.clients)) if self.adj[ci][cj] != np.inf]
        self.G = nx.Graph()
        for node in self.nodes:
            self.G.add_node(node)
        self.G.add_edges_from(self.edges)
        self.colors = ['r', 'b', 'y', 'g', 'black', 'pink', 'orange', 'gold', 'lightskyblue', '#ffb3e6']
        self.output['sampled_times'] = [0 for _ in range(len(self.clients))]


    def log_per_round(self, *args, **kwargs):
        test_metric = self.server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        valid_metrics = self.server.test_on_clients('valid')
        for met_name, met_val in valid_metrics.items():
            self.output['valid_'+met_name+'_dist'].append(met_val)
            self.output['valid_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(self.server.local_data_vols, met_val)]) / self.server.total_data_vol)
            self.output['mean_valid_' + met_name].append(np.mean(met_val))
            self.output['std_valid_' + met_name].append(np.std(met_val))
        # active mode
        self.output['active_mode'].append([c.active for c in self.clients])
        tmp = [int(sid) for sid in self.server.selected_clients]
        self.output['selected_clients'].append(tmp)
        for cid in self.server.selected_clients:
            self.output['sampled_times'][cid] += 1
        # node_labels = [str(si) for si in self.sample_times]
        # self.draw_graph_with_pie(self.client_features, colors=self.colors, pie_size=0.04, node_labels=node_labels, label_delta=[-0.07, 0.07], with_selected=True)
        # output to stdout
        self.show_current_output()

    def organize_output(self, *args, **kwargs):
        self.output['meta'] = self.meta
        for key in self.output.keys():
            if '_dist' in key:
                self.output[key] = self.output[key][-1]
        num_samples = [0 for _ in self.clients]
        for r in self.output['selected_clients']:
            for cid in r:
                num_samples[cid]+=1
        self.output['sampled_time_dist'] = num_samples
        # sort active mode for clients
        a = np.array(self.output['active_mode'], dtype=int)
        a = a[1:]
        sa = np.sum(a.T, axis=1)
        sorted_clients = sa.argsort().tolist()
        idx = {s:i for s,i in zip(sorted_clients, [i for i in range(len(self.clients))])}
        # active_mode
        tmp = []
        for cid in range(len(self.clients)):
            for r in range(1, len(self.output['active_mode'])):
                if self.output['active_mode'][r][cid]:
                    tmp.append([r - 1, idx[cid]])
        self.output['active_mode'] = tmp
        # selected_clients

        selected_clients = []
        for r in range(1, len(self.output['selected_clients'])):
            for cid in self.output['selected_clients'][r]:
                selected_clients.append([r-1, idx[cid]])
        self.output['selected_clients'] = selected_clients
        return

    def get_client_feature(self):
        if self.meta['task'].startswith('distributedQP'):
            return
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
            return client_features

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
                    if nid in self.server.selected_clients:
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
                if nid in self.server.selected_clients:
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