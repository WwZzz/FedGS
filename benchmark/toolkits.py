"""
DISTRIBUTION OF DATASET
-----------------------------------------------------------------------------------
balance:
    iid:            0 : identical and independent distributions of the dataset among clients
    label skew:     1 Quantity:  each party owns data samples of a fixed number of labels.
                    2 Dirichlet: each party is allocated a proportion of the samples of each label according to Dirichlet distribution.
                    3 Shard: each party is allocated the same numbers of shards that is sorted by the labels of the data
                    4 Dirichlet-Imbalance: the label distribution is the same to dist2 but the local data sizes vary a lot.
-----------------------------------------------------------------------------------
depends on partitions:
    feature skew:
                    5 ID: For Shakespeare\FEMNIST, we divide and assign the writers (and their characters) into each party randomly and equally.
-----------------------------------------------------------------------------------
imbalance:
    iid:            6 Vol: only the vol of local dataset varies.
    niid:           7 Vol: for generating synthetic_regression data
"""

import ujson
import shutil
import numpy as np
import os.path
import random
import os
import ssl
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import torch
ssl._create_default_https_context = ssl._create_unverified_context
import importlib
import collections
from torchvision import datasets, transforms
import utils.fmodule

# ========================================Task Generator============================================
# This part is for generating federated dataset from original dataset. The generation process should be
# implemented in the method run(). Here we provide a basic class BasicTaskGen as a standard process to
# generate federated dataset, which mainly includes:
#   1) loading and pre-processing data by load_data(),
#   2) partitioning dataset by partition(),
#   3) saving the partitioned dataset for clients and test dataset for server as the final fedtask by save_data().
# We also provide a default task generator DefaultTaskGen to cover the generating of several datasets (e.g. MNIST,
# CIFAR10, CIFAR100, FashionMNIST, EMNIST)， which enables joining different datasets with very few code (please see
# the core.py at the path of these datasets for details).
class BasicTaskGen:
    _TYPE_DIST = {
        0: 'iid',
        1: 'label_skew_quantity',
        2: 'label_skew_dirichlet',
        3: 'label_skew_shard',
        4: 'feature_skew_noise',
        5: 'feature_skew_id',
        6: 'iid_volumn_skew',
        7: 'niid_volumn_skew',
        8: 'concept skew',
        9: 'concept and feature skew and balance',
        10: 'concept and feature skew and imbalance',
    }
    _TYPE_DATASET = ['2DImage', '3DImage', 'Text', 'Sequential', 'Graph', 'Tabular']

    def __init__(self, benchmark, dist_id, skewness, rawdata_path, local_hld_rate=0.2, seed=0):
        """
        :param benchmark: the name of the ML task to be converted
        :param dist_id:  the index for ensuring the type of the data distribution
        :param skewness: the degree of data heterogeneity which grows with skewness from 0 to 1
        :param rawdata_path: the dictionary of the original dataset
        :param local_hld_rate: the hold-out rate of each local dataset (e.g. |valid_data| = |all_data|*local_hld_rate)
        :param seed: random seed
        """
        self.benchmark = benchmark
        self.task_rootpath = './fedtask'
        self.rawdata_path = rawdata_path
        self.dist_id = dist_id
        self.dist_name = self._TYPE_DIST[dist_id]
        self.skewness = 0 if dist_id==0 else skewness
        self.num_clients = -1
        self.local_holdout_rate = local_hld_rate
        self.seed = seed
        self.set_random_seed(self.seed)

    def run(self, *args, **kwargs):
        """The whole process to generate federated task. """
        pass

    def load_data(self, *args, **kwargs):
        """Download and load dataset into memory."""
        pass

    def partition(self, *args, **kwargs):
        """Partition the data according to 'dist' and 'skewness'"""
        pass

    def save_task(self, *args, **kwargs):
        """Save the federated dataset to the task_path/data.
        This algorithm should be implemented as the way to read
        data from disk that is defined in TaskPipe.read_task()
        """
        pass

    def local_holdout(self, *args, **kwargs):
        """hold-out the validation dataset from each local dataset"""
        pass

    def save_info(self, *args, **kwargs):
        """Save the task infomation to the .json file stored in taskpath"""
        pass

    def get_taskname(self):
        """Create task name and return it."""
        taskname = '_'.join([self.benchmark, 'cnum' +  str(self.num_clients), 'dist' + str(self.dist_id), 'skew' + str(self.skewness).replace(" ", ""), 'seed'+str(self.seed)])
        return taskname

    def get_client_names(self):
        k = str(len(str(self.num_clients)))
        return [('Client{:0>' + k + 'd}').format(i) for i in range(self.num_clients)]

    def create_task_directories(self):
        """Create the directories of the task."""
        taskname = self.get_taskname()
        taskpath = os.path.join(self.task_rootpath, taskname)
        os.mkdir(taskpath)
        os.mkdir(os.path.join(taskpath, 'record'))
        os.mkdir(os.path.join(taskpath, 'log'))

    def _check_task_exist(self):
        """Check whether the task already exists."""
        taskname = self.get_taskname()
        return os.path.exists(os.path.join(self.task_rootpath, taskname))

    def set_random_seed(self,seed=0):
        """Set random seed"""
        random.seed(3 + seed)
        np.random.seed(97 + seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _remove_task(self):
        "remove the task when generating failed"
        if self._check_task_exist():
            taskname = self.get_taskname()
            taskpath = os.path.join(self.task_rootpath, taskname)
            shutil.rmtree(taskpath)
        return

class DefaultTaskGen(BasicTaskGen):
    def __init__(self, benchmark, dist_id, skewness, rawdata_path, num_clients=1, minvol=10, local_hld_rate=0.2, seed=0):
        super(DefaultTaskGen, self).__init__(benchmark, dist_id, skewness, rawdata_path, local_hld_rate, seed)
        self.minvol=minvol
        self.num_classes = -1
        self.train_data = None
        self.test_data = None
        self.num_clients = num_clients
        self.cnames = self.get_client_names()
        self.taskname = self.get_taskname()
        self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        self.visualize = None
        self.source_dict = {
            'lib': None,
            'class_name': None,
            'train_args': {},
            'test_args': {},
        }

    def run(self):
        """ Generate federated task"""
        # check if the task exists
        if self._check_task_exist():
            print("Task Already Exists.")
            return
        # read raw_data into self.train_data and self.test_data
        print('-----------------------------------------------------')
        print('Loading...')
        self.load_data()
        print('Done.')
        # partition data and hold-out for each local dataset
        print('-----------------------------------------------------')
        print('Partitioning data...')
        local_datas = self.partition()
        self.train_cidxs, self.valid_cidxs = self.local_holdout(local_datas, shuffle=True)
        print('Done.')
        # save task infomation as .json file and the federated dataset
        print('-----------------------------------------------------')
        print('Saving data...')
        try:
            # create the directory of the task
            self.create_task_directories()
            # visualizing partition
            if self.visualize is not None:
                self.visualize(self.train_cidxs)
            self.save_task(self)
        except:
            self._remove_task()
            print("Failed to saving splited dataset.")
        print('Done.')
        return

    def load_data(self):
        """ load and pre-process the raw data"""
        return

    def partition(self):
        # Partition self.train_data according to the delimiter and return indexes of data owned by each client as [c1data_idxs, ...] where the type of each element is list(int)
        if self.dist_id == 0:
            """IID"""
            d_idxs = np.random.permutation(len(self.train_data))
            local_datas = np.array_split(d_idxs, self.num_clients)
            local_datas = [data_idx.tolist() for data_idx in local_datas]

        elif self.dist_id == 1:
            """label_skew_quantity"""
            dpairs = [[did, self.train_data[did][-1]] for did in range(len(self.train_data))]
            num = max(int((1-self.skewness)*self.num_classes), 1)
            K = self.num_classes
            local_datas = [[] for _ in range(self.num_clients)]
            if num == K:
                for k in range(K):
                    idx_k = [p[0] for p in dpairs if p[1]==k]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, self.num_clients)
                    for cid in range(self.num_clients):
                        local_datas[cid].extend(split[cid].tolist())
            else:
                times = [0 for _ in range(self.num_classes)]
                contain = []
                for i in range(self.num_clients):
                    current = []
                    j =0
                    while (j < num):
                        mintime = np.min(times)
                        ind = np.random.choice(np.where(times == mintime)[0])
                        if (ind not in current):
                            j = j + 1
                            current.append(ind)
                            times[ind] += 1
                    contain.append(current)
                for k in range(K):
                    idx_k = [p[0] for p in dpairs if p[1]==k]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, times[k])
                    ids = 0
                    for cid in range(self.num_clients):
                        if k in contain[cid]:
                            local_datas[cid].extend(split[ids].tolist())
                            ids += 1

        elif self.dist_id == 2:
            """label_skew_dirichlet"""
            """alpha = (-4log(skewness + epsilon))**4"""
            MIN_ALPHA = 0.01
            alpha = (-4*np.log(self.skewness + 10e-8))**4
            alpha = max(alpha, MIN_ALPHA)
            labels = [self.train_data[did][-1] for did in range(len(self.train_data))]
            lb_counter = collections.Counter(labels)
            p = np.array([1.0*v/len(self.train_data) for v in lb_counter.values()])
            lb_dict = {}
            labels = np.array(labels)
            for lb in range(len(lb_counter.keys())):
                lb_dict[lb] = np.where(labels==lb)[0]
            proportions = [np.random.dirichlet(alpha*p) for _ in range(self.num_clients)]
            while np.any(np.isnan(proportions)):
                proportions = [np.random.dirichlet(alpha * p) for _ in range(self.num_clients)]
            while True:
                # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
                mean_prop = np.mean(proportions, axis=0)
                error_norm = ((mean_prop-p)**2).sum()
                print("Error: {:.8f}".format(error_norm))
                if error_norm<=1e-2/self.num_classes:
                    break
                exclude_norms = []
                for cid in range(self.num_clients):
                    mean_excid = (mean_prop*self.num_clients-proportions[cid])/(self.num_clients-1)
                    error_excid = ((mean_excid-p)**2).sum()
                    exclude_norms.append(error_excid)
                excid = np.argmin(exclude_norms)
                sup_prop = [np.random.dirichlet(alpha*p) for _ in range(self.num_clients)]
                alter_norms = []
                for cid in range(self.num_clients):
                    if np.any(np.isnan(sup_prop[cid])):
                        continue
                    mean_alter_cid = mean_prop - proportions[excid]/self.num_clients + sup_prop[cid]/self.num_clients
                    error_alter = ((mean_alter_cid-p)**2).sum()
                    alter_norms.append(error_alter)
                if len(alter_norms)>0:
                    alcid = np.argmin(alter_norms)
                    proportions[excid] = sup_prop[alcid]
            local_datas = [[] for _ in range(self.num_clients)]
            self.dirichlet_dist = [] # for efficiently visualizing
            for lb in lb_counter.keys():
                lb_idxs = lb_dict[lb]
                lb_proportion = np.array([pi[lb] for pi in proportions])
                lb_proportion = lb_proportion/lb_proportion.sum()
                lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
                lb_datas = np.split(lb_idxs, lb_proportion)
                self.dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
                local_datas = [local_data+lb_data.tolist() for local_data,lb_data in zip(local_datas, lb_datas)]
            self.dirichlet_dist = np.array(self.dirichlet_dist).T
            for i in range(self.num_clients):
                np.random.shuffle(local_datas[i])

        elif self.dist_id == 3:
            """label_skew_shard"""
            dpairs = [[did, self.train_data[did][-1]] for did in range(len(self.train_data))]
            self.skewness = min(max(0, self.skewness), 1.0)
            num_shards = max(int((1 - self.skewness) * self.num_classes * 2), 1)
            client_datasize = int(len(self.train_data) / self.num_clients)
            all_idxs = [i for i in range(len(self.train_data))]
            z = zip([p[1] for p in dpairs], all_idxs)
            z = sorted(z)
            labels, all_idxs = zip(*z)
            shardsize = int(client_datasize / num_shards)
            idxs_shard = range(int(self.num_clients * num_shards))
            local_datas = [[] for i in range(self.num_clients)]
            for i in range(self.num_clients):
                rand_set = set(np.random.choice(idxs_shard, num_shards, replace=False))
                idxs_shard = list(set(idxs_shard) - rand_set)
                for rand in rand_set:
                    local_datas[i].extend(all_idxs[rand * shardsize:(rand + 1) * shardsize])

        elif self.dist_id == 4:
            """label_skew_dirichlet with imbalance data size. The data"""
            # calculate alpha = (-4log(skewness + epsilon))**4
            MIN_ALPHA = 0.01
            alpha = (-4 * np.log(self.skewness + 10e-8)) ** 4
            alpha = max(alpha, MIN_ALPHA)
            # ensure imbalance data sizes
            total_data_size = len(self.train_data)
            mean_datasize = total_data_size/self.num_clients
            mu = np.log(mean_datasize) - 0.5
            sigma = 1
            samples_per_client = np.random.lognormal(mu, sigma, (self.num_clients)).astype(int)
            thresold = int(0.1*total_data_size)
            delta = int(0.1 * thresold)
            crt_data_size = sum(samples_per_client)
            # force current data size to match the total data size
            while crt_data_size != total_data_size:
                if crt_data_size - total_data_size >= thresold:
                    maxid = np.argmax(samples_per_client)
                    samples = np.random.lognormal(mu, sigma, (self.num_clients))
                    new_size_id = np.argmin([np.abs(crt_data_size-samples_per_client[maxid]+s) for s in samples])
                    samples_per_client[maxid] = samples[new_size_id]
                elif crt_data_size - total_data_size >= delta:
                    maxid = np.argmax(samples_per_client)
                    samples_per_client[maxid] -= delta
                elif crt_data_size - total_data_size >0:
                    maxid = np.argmax(samples_per_client)
                    samples_per_client[maxid] -= (crt_data_size - total_data_size)
                elif total_data_size - crt_data_size >= delta:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += delta
                elif total_data_size - crt_data_size >= delta:
                    minid = np.argmin(samples_per_client)
                    samples = np.random.lognormal(mu, sigma, (self.num_clients))
                    new_size_id = np.argmin([np.abs(crt_data_size-samples_per_client[minid]+s) for s in samples])
                    samples_per_client[minid] = samples[new_size_id]
                else:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += (total_data_size - crt_data_size)
                crt_data_size = sum(samples_per_client)
            # count the label distribution
            labels = [self.train_data[did][-1] for did in range(len(self.train_data))]
            lb_counter = collections.Counter(labels)
            p = np.array([1.0 * v / len(self.train_data) for v in lb_counter.values()])
            lb_dict = {}
            labels = np.array(labels)
            for lb in range(len(lb_counter.keys())):
                lb_dict[lb] = np.where(labels == lb)[0]
            proportions = [np.random.dirichlet(alpha * p) for _ in range(self.num_clients)]
            while np.any(np.isnan(proportions)):
                proportions = [np.random.dirichlet(alpha * p) for _ in range(self.num_clients)]
            sorted_cid_map = {k:i for k,i in zip(np.argsort(samples_per_client), [_ for _ in range(self.num_clients)])}
            crt_id = 0
            while True:
                # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
                mean_prop = np.sum([pi*di for pi,di in zip(proportions, samples_per_client)], axis=0)
                mean_prop = mean_prop/mean_prop.sum()
                error_norm = ((mean_prop - p) ** 2).sum()
                print("Error: {:.8f}".format(error_norm))
                if error_norm <= 1e-2 / self.num_classes:
                    break
                excid = sorted_cid_map[crt_id]
                crt_id = (crt_id+1)%self.num_clients
                sup_prop = [np.random.dirichlet(alpha * p) for _ in range(self.num_clients)]
                del_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
                del_prop -= samples_per_client[excid]*proportions[excid]
                alter_norms = []
                for cid in range(self.num_clients):
                    if np.any(np.isnan(sup_prop[cid])):
                        continue
                    alter_prop = del_prop + samples_per_client[excid]*sup_prop[cid]
                    alter_prop = alter_prop/alter_prop.sum()
                    error_alter = ((alter_prop - p) ** 2).sum()
                    alter_norms.append(error_alter)
                if len(alter_norms) > 0:
                    alcid = np.argmin(alter_norms)
                    proportions[excid] = sup_prop[alcid]
            local_datas = [[] for _ in range(self.num_clients)]
            self.dirichlet_dist = []  # for efficiently visualizing
            for lb in lb_counter.keys():
                lb_idxs = lb_dict[lb]
                lb_proportion = np.array([pi[lb]*si for pi,si in zip(proportions, samples_per_client)])
                lb_proportion = lb_proportion / lb_proportion.sum()
                lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
                lb_datas = np.split(lb_idxs, lb_proportion)
                self.dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
                local_datas = [local_data + lb_data.tolist() for local_data, lb_data in zip(local_datas, lb_datas)]
            self.dirichlet_dist = np.array(self.dirichlet_dist).T
            for i in range(self.num_clients):
                np.random.shuffle(local_datas[i])

        elif self.dist_id == 5:
            """feature_skew_id"""
            if not isinstance(self.train_data, TupleDataset):
                raise RuntimeError("Support for dist_id=5 only after setting the type of self.train_data is TupleDataset")
            Xs, IDs, Ys = self.train_data.tolist()
            self.num_clients = len(set(IDs))
            local_datas = [[] for _ in range(self.num_clients)]
            for did in range(len(IDs)):
                local_datas[IDs[did]].append(did)

        elif self.dist_id == 6:
            minv = 0
            d_idxs = np.random.permutation(len(self.train_data))
            while minv < self.minvol:
                proportions = np.random.dirichlet(np.repeat(self.skewness, self.num_clients))
                proportions = proportions / proportions.sum()
                minv = np.min(proportions * len(self.train_data))
            proportions = (np.cumsum(proportions) * len(d_idxs)).astype(int)[:-1]
            local_datas  = np.split(d_idxs, proportions)
        return local_datas

    def local_holdout(self, local_datas, shuffle=False):
        """split each local dataset into train data and valid data according the rate."""
        train_cidxs = []
        valid_cidxs = []
        for local_data in local_datas:
            if shuffle:
                np.random.shuffle(local_data)
            k = int(len(local_data) * (1-self.local_holdout_rate))
            train_cidxs.append(local_data[:k])
            valid_cidxs.append(local_data[k:])
        return train_cidxs, valid_cidxs

    def convert_data_for_saving(self):
        """Convert self.train_data and self.test_data to list that can be stored as .json file and the converted dataset={'x':[], 'y':[]}"""
        pass

    def visualize_by_class(self, train_cidxs):
        import collections
        import matplotlib.pyplot as plt
        import matplotlib.colors
        import random
        ax = plt.subplots()
        colors = [key for key in matplotlib.colors.CSS4_COLORS.keys()]
        random.shuffle(colors)
        client_height = 1
        if hasattr(self, 'dirichlet_dist'):
            client_dist = self.dirichlet_dist.tolist()
            data_columns = [sum(cprop) for cprop in client_dist]
            row_map = {k:i for k,i in zip(np.argsort(data_columns), [_ for _ in range(self.num_clients)])}
            for cid, cprop in enumerate(client_dist):
                offset = 0
                y_bottom = row_map[cid] - client_height/2.0
                y_top = row_map[cid] + client_height/2.0
                for lbi in range(len(cprop)):
                    plt.fill_between([offset,offset+cprop[lbi]], y_bottom, y_top, facecolor = colors[lbi])
                    # plt.barh(cid, cprop[lbi], client_height, left=offset, color=)
                    offset += cprop[lbi]
        else:
            data_columns = [len(cidx) for cidx in train_cidxs]
            row_map = {k:i for k,i in zip(np.argsort(data_columns), [_ for _ in range(self.num_clients)])}
            for cid, cidxs in enumerate(train_cidxs):
                labels = [int(self.train_data[did][-1]) for did in cidxs]
                lb_counter = collections.Counter(labels)
                offset = 0
                y_bottom = row_map[cid] - client_height/2.0
                y_top = row_map[cid] + client_height/2.0
                for lbi in range(self.num_classes):
                    plt.fill_between([offset,offset+lb_counter[lbi]], y_bottom, y_top, facecolor = colors[lbi])
                    offset += lb_counter[lbi]
        plt.xlim(0,max(data_columns))
        plt.ylim(-0.5,len(train_cidxs)-0.5)
        plt.ylabel('Client ID')
        plt.xlabel('Number of Samples')
        plt.title(self.get_taskname())
        plt.savefig(os.path.join(self.taskpath, self.get_taskname()+'.jpg'))
        plt.show()

# =======================================Task Calculator===============================================
# This module is to seperate the task-specific calculating part from the federated algorithms, since the
# way of calculation (e.g. loss, evaluating metrics, optimizer) and the format of data (e.g. image, text)
# can vary in different dataset. Therefore, this module should provide a standard interface for the federated
# algorithms. To achieve this, we list the necessary interfaces as follows:
#   1) data_to_device: put the data into cuda device, since different data may differ in size or shape.
#   2) get_data_loader: get the data loader which is enumerable and returns a batch of data
#   3) get_optimizer: get the optimizer for optimizing the model parameters, which can also vary among different datasets
#   4) get_loss: the basic loss calculating procedure for the dataset, and returns loss as the final point of the computing graph
#   5) get_evaluation: evaluating the model on the dataset
# The same as TaskGenerator, we provide a default task calculator ClassifyCalculator that is suitable for datasets
# like MNIST, CIFAR100.
class BasicTaskCalculator:
    def __init__(self, device, optimizer_name='sgd'):
        self.device = device
        self.optimizer_name = optimizer_name
        self.criterion = None
        self.DataLoader = None

    def data_to_device(self, data):
        raise NotImplementedError

    def train_one_step(self, *args, **kwargs):
        raise NotImplementedError

    def get_evaluation(self, *args, **kwargs):
        raise NotImplementedError

    def get_data_loader(self, data, batch_size=64, shuffle=True):
        return NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def get_optimizer(self, model=None, lr=0.1, weight_decay=0, momentum=0):
        # if self._OPTIM == None:
        #     raise RuntimeError("TaskCalculator._OPTIM Not Initialized.")
        if self.optimizer_name.lower() == 'sgd':
            return getattr(importlib.import_module('torch.optim'), self.optimizer_name)(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            # return utils.fmodule.Optim(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif self.optimizer_name.lower() == 'adam':
            return getattr(importlib.import_module('torch.optim'), self.optimizer_name)(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay, amsgrad=True)
            # return utils.fmodule.Optim(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay, amsgrad=True)
        else:
            raise RuntimeError("Invalid Optimizer.")

    @classmethod
    def setOP(cls, OP):
        cls._OPTIM = OP

class ClassificationCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(ClassificationCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.DataLoader = DataLoader

    def train_one_step(self, model, data):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.data_to_device(data)
        outputs = model(tdata[0])
        loss = self.criterion(outputs, tdata[-1])
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        num_correct = 0
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            outputs = model(batch_data[0])
            batch_mean_loss = self.criterion(outputs, batch_data[-1]).item()
            y_pred = outputs.data.max(1, keepdim=True)[1]
            correct = y_pred.eq(batch_data[-1].data.view_as(y_pred)).long().cpu().sum()
            num_correct += correct.item()
            total_loss += batch_mean_loss * len(batch_data[-1])
        return {'accuracy': 1.0*num_correct/len(dataset), 'loss':total_loss/len(dataset)}

    def data_to_device(self, data):
        return data[0].to(self.device), data[1].to(self.device)

    def get_data_loader(self, dataset, batch_size=64, shuffle=True, num_workers=0):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# =====================================Task Reader\xxDataset======================================================
# This module is to read the fedtask that is generated by Generator. The target is to load the fedtask into a
# dataset inheriting from torch.utils.data.Dataset. Thus the only method read_data() should be rewriten to be suitable
# for the corresponding generating manner. With the consideration of various shapes of data, we provide mainly two ways
# for saving data and reading data:
#   1) Save the partitioned indices of items in the original dataset (e.g. torch.torchvision.*) and the path of
#      the original dataset into .json file. Then dynamically importing the original dataset when running federated training procedure,
#      and specifying each local dataset by the local indices. This function is implemented by IDXDataset and IDXTaskReader.
#      The advantages of this way include saving storing memory, high efficiency and the full usage of the torch implemention of
#      datasets in torchvision and torchspeech. Examples can be found in mnist_classification\core.py, cifar\core.py.
#
#   2) Save the partitioned data itself into .json file. Then read the data. The advantage of this way is the flexibility.
#      Examples can be found in emnist_classification\core.py, synthetic_regression\core.py, distributedQP\core.py.

class BasicTaskPipe:
    """
    A Pipe for saving the partitioned dataset as .json file (i.e. fedtask)
    and reading the stored fedtask into the federated system.

        TaskPipe.save_task: the operation of saving a task should be complemented here
        TaskPipe.load_task: the operation of loading a task should be complemented here
        TaskPipe.TaskDataset: when running main.py to start the training procedure, each
                          dataset should be loaded with this type of class (i.e. server.test_data
                          client.train_data, client.valid_data)
    """
    TaskDataset = None

    @classmethod
    def load_task(cls, *args, **kwargs):
        """
            Reading the spilted dataset from disk files and loading data into the class 'LocalDataset'.
            This algorithm should read three types of data from the processed task:
                train_sets = [client1_train_data, ...] where each item is an instance of 'LocalDataset'
                valid_sets = [client1_valid_data, ...] where each item is an instance of 'LocalDataset'
                test_set = test_dataset
            Return train_sets, valid_sets, test_set, client_names
        """
        raise NotImplementedError

    @classmethod
    def save_task(cls, *args, **kwargs):
        """save the federated task as .json file"""
        raise NotImplementedError

class XYTaskPipe(BasicTaskPipe):
    class XYDataset(Dataset):
        def __init__(self, X=[], Y=[], totensor=True):
            """ Init Dataset with pairs of features and labels/annotations.
            XYDataset transforms data that is list\array into tensor.
            The data is already loaded into memory before passing into XYDataset.__init__()
            and thus is only suitable for benchmarks with small size (e.g. CIFAR10, MNIST)
            Args:
                X: a list of features
                Y: a list of labels with the same length of X
            """
            if not self._check_equal_length(X, Y):
                raise RuntimeError("Different length of Y with X.")
            if totensor:
                try:
                    self.X = torch.tensor(X)
                    self.Y = torch.tensor(Y)
                except:
                    raise RuntimeError("Failed to convert input into torch.Tensor.")
            else:
                self.X = X
                self.Y = Y
            self.all_labels = list(set(self.tolist()[1]))

        def __len__(self):
            return len(self.Y)

        def __getitem__(self, item):
            return self.X[item], self.Y[item]

        def tolist(self):
            if not isinstance(self.X, torch.Tensor):
                return self.X, self.Y
            return self.X.tolist(), self.Y.tolist()

        def _check_equal_length(self, X, Y):
            return len(X) == len(Y)

        def get_all_labels(self):
            return self.all_labels
    TaskDataset = XYDataset
    @classmethod
    def save_task(cls, generator):
        """
        Store all the features (i.e. X) and coresponding labels (i.e. Y) into disk as .json file.
        The input 'generator' must have attributes:
            :taskpath: string. the path of storing
            :train_data: the training dataset which is a dict {'x':..., 'y':...}
            :test_data: the testing dataset which is a dict {'x':..., 'y':...}
            :train_cidxs: a list of lists of integer. The splited indices in train_data of the training part of each local dataset
            :valid_cidxs: a list of lists of integer. The splited indices in train_data of the valiadtion part of each local dataset
            :client_names: a list of strings. The names of all the clients, which is used to index the clients' data in .json file
            :return:
        """
        feddata = {
            'store': 'XY',
            'client_names': generator.cnames,
            'dtest': generator.test_data
        }
        for cid in range(len(generator.cnames)):
            feddata[generator.cnames[cid]] = {
                'dtrain': {
                    'x': [generator.train_data['x'][did] for did in generator.train_cidxs[cid]],
                    'y': [generator.train_data['y'][did] for did in generator.train_cidxs[cid]]
                },
                'dvalid': {
                    'x': [generator.train_data['x'][did] for did in generator.valid_cidxs[cid]],
                    'y': [generator.train_data['y'][did] for did in generator.valid_cidxs[cid]]
                }
            }
        with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)

    @classmethod
    def load_task(cls, task_path):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        test_data = cls.TaskDataset(feddata['dtest']['x'], feddata['dtest']['y'])
        train_datas = [cls.TaskDataset(feddata[name]['dtrain']['x'], feddata[name]['dtrain']['y']) for name in feddata['client_names']]
        valid_datas = [cls.TaskDataset(feddata[name]['dvalid']['x'], feddata[name]['dvalid']['y']) for name in feddata['client_names']]
        return train_datas, valid_datas, test_data, feddata['client_names']

class IDXTaskPipe(BasicTaskPipe):
    # class IDXDataset(Dataset):
    #     # The source dataset that can be indexed by IDXDataset
    #     _ORIGIN_DATA = {'TRAIN': None, 'TEST': None, 'CLASS': None}
    #
    #     def __init__(self, idxs, key='TRAIN'):
    #         """Init dataset with 'src_data' and a list of indexes that are used to position data in 'src_data'"""
    #         if not isinstance(idxs, list):
    #             raise RuntimeError("Invalid Indexes")
    #         self.idxs = idxs
    #         self.key = key
    #
    #     @classmethod
    #     def SET_ORIGIN_DATA(cls, train_data=None, test_data=None):
    #         cls._ORIGIN_DATA['TRAIN'] = train_data
    #         cls._ORIGIN_DATA['TEST'] = test_data
    #
    #     @classmethod
    #     def SET_ORIGIN_CLASS(cls, DataClass=None):
    #         cls._ORIGIN_DATA['CLASS'] = DataClass
    #
    #     @classmethod
    #     def ADD_KEY_TO_DATA(cls, key, value=None):
    #         if key == None:
    #             raise RuntimeError("Empty key when calling class algorithm IDXData.ADD_KEY_TO_DATA")
    #         cls._ORIGIN_DATA[key] = value
    #
    #     def __getitem__(self, item):
    #         idx = self.idxs[item]
    #         return self._ORIGIN_DATA[self.key][idx]
    #
    #     def __len__(self):
    #         return len(self.idxs)

    TaskDataset = Subset
    @classmethod
    def save_task(cls, generator):
        """
        Store the splited indices of the local data in the original dataset (source dataset) into the disk as .json file
        The input 'generator' must have attributes:
            :taskpath: string. the path of storing
            :train_data: the training dataset which is a dict {'x':..., 'y':...}
            :test_data: the testing dataset which is a dict {'x':..., 'y':...}
            :train_cidxs: a list of lists of integer. The splited indices in train_data of the training part of each local dataset
            :valid_cidxs: a list of lists of integer. The splited indices in train_data of the valiadtion part of each local dataset
            :client_names: a list of strings. The names of all the clients, which is used to index the clients' data in .json file
            :source_dict: a dict that contains parameters which is necessary to dynamically importing the original Dataset class and generating instances
                    For example, for MNIST using this task pipe, the source_dict should be like:
                    {'class_path': 'torchvision.datasets',
                     'class_name': 'MNIST',
                     'train_args': {'root': '"'+MNIST_rawdata_path+'"', 'download': 'True', 'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])','train': 'True'},
                     'test_args': {'root': '"'+MNIST_rawdata_path+'"', 'download': 'True', 'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])', 'train': 'False'}
                    }
            :return:
        """
        feddata = {
            'store': 'IDX',
            'client_names': generator.cnames,
            'dtest': [i for i in range(len(generator.test_data))],
            'datasrc': generator.source_dict
        }
        for cid in range(len(generator.cnames)):
            feddata[generator.cnames[cid]] = {
                'dtrain': generator.train_cidxs[cid],
                'dvalid': generator.valid_cidxs[cid]
            }
        with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return

    @classmethod
    def load_task(cls, task_path):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        class_path = feddata['datasrc']['class_path']
        class_name = feddata['datasrc']['class_name']
        origin_class = getattr(importlib.import_module(class_path), class_name)
        origin_train_data = cls.args_to_dataset(origin_class, feddata['datasrc']['train_args'])
        origin_test_data = cls.args_to_dataset(origin_class, feddata['datasrc']['test_args'])
        test_data = cls.TaskDataset(origin_test_data, [_ for _ in range(len(origin_test_data))])
        train_datas = [cls.TaskDataset(origin_train_data, feddata[name]['dtrain']) for name in feddata['client_names']]
        valid_datas = [cls.TaskDataset(origin_train_data, feddata[name]['dvalid']) for name in feddata['client_names']]
        return train_datas, valid_datas, test_data, feddata['client_names']

    @classmethod
    def args_to_dataset(cls, original_class, args):
        if not isinstance(args, dict):
            raise TypeError
        args_str = '(' + ','.join([key + '=' + value for key, value in args.items()]) + ')'
        return eval("original_class" + args_str)

class XTaskPipe(BasicTaskPipe):
    class XDataset(Dataset):
        def __init__(self, X=[], totensor=True):
            if totensor:
                try:
                    self.X = torch.tensor(X)
                except:
                    raise RuntimeError("Failed to convert input into torch.Tensor.")
            else:
                self.X = X

        def __getitem__(self, item):
            return self.X[item]

        def __len__(self):
            return len(self.X)

    TaskDataset = XDataset
    @classmethod
    def save_task(cls, generator):
        """
        Store all the features (i.e. X) into the disk as .json file.
        The input 'generator' must have attributes:
            :taskpath: string. the path of storing
            :train_data: the training dataset which is a dict {'x':..., 'y':...}
            :test_data: the testing dataset which is a dict {'x':..., 'y':...}
            :train_cidxs: a list of lists of integer. The splited indices in train_data of the training part of each local dataset
            :valid_cidxs: a list of lists of integer. The splited indices in train_data of the valiadtion part of each local dataset
            :client_names: a list of strings. The names of all the clients, which is used to index the clients' data in .json file
            :return:
        """
        feddata = {
            'store': 'X',
            'client_names': generator.cnames,
            'dtest': generator.test_data
        }
        for cid in range(generator.num_clients):
            feddata[generator.cnames[cid]] = {
                'dtrain': {
                    'x': [generator.train_data[did] for did in generator.train_cidxs[cid]],
                },
                'dvalid': {
                    'x': [generator.train_data[did] for did in generator.valid_cidxs[cid]],
                }
            }
        with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return

    @classmethod
    def load_task(cls, task_path):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        test_data = cls.TaskDataset(feddata['dtest']['x'])
        train_datas = [cls.TaskDataset(feddata[name]['dtrain']['x']) for name in feddata['client_names']]
        valid_datas = [cls.TaskDataset(feddata[name]['dvalid']['x']) for name in feddata['client_names']]
        return train_datas, valid_datas, test_data, feddata['client_names']

class TupleDataset(Dataset):
    def __init__(self, X1=[], X2=[], Y=[], totensor=True):
        if totensor:
            try:
                self.X1 = torch.tensor(X1)
                self.X2 = torch.tensor(X2)
                self.Y = torch.tensor(Y)
            except:
                raise RuntimeError("Failed to convert input into torch.Tensor.")
        else:
            self.X1 = X1
            self.X2 = X2
            self.Y = Y

    def __getitem__(self, item):
        return self.X1[item], self.X2[item], self.Y[item]

    def __len__(self):
        return len(self.Y)

    def tolist(self):
        if not isinstance(self.X1, torch.Tensor):
            return self.X1, self.X2, self.Y
        return self.X1.tolist(), self.X2.tolist(), self.Y.tolist()

