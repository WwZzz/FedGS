from torchvision import datasets, transforms
from benchmark.toolkits import ClassificationCalculator, DefaultTaskGen,BasicTaskPipe
import os
import warnings
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
import numpy as np
import ujson
import importlib

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, seed = 0):
        super(TaskGen, self).__init__('domain_digit_classification', dist_id=dist_id, num_clients=num_clients, skewness=skewness, rawdata_path='./benchmark/RAW_DATA', seed=seed)
        self.domain_names = ['SVHN', 'MNIST', 'USPS', 'SyntheticDigits', 'MNISTM']
        self.domain_info = {
            'SVHN':{'class': SVHN, 'trans': None, 'train_data': None, 'test_data': None, 'num_clients':0, 'class_path': 'benchmark.domain_digit_classification.core', 'trans':'transforms.Compose([transforms.Resize([28,28]),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])'},
            'MNIST': {'class': datasets.MNIST, 'trans': None, 'train_data': None, 'test_data': None, 'num_clients':0,  'class_path': 'torchvision.datasets', 'trans':'transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])'},
            'USPS':{'class': datasets.USPS, 'trans': None, 'train_data': None, 'test_data': None, 'num_clients':0, 'class_path': 'torchvision.datasets','trans':'transforms.Compose([transforms.Resize([28,28]),transforms.Grayscale(num_output_channels=3),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])'},
            'SyntheticDigits':{'class': SyntheticDigits, 'trans': None, 'train_data': None, 'test_data': None, 'num_clients':0,  'class_path': 'benchmark.domain_digit_classification.core','trans':'transforms.Compose([transforms.Resize([28,28]),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])'},
            'MNISTM':{'class': MNISTM, 'trans':None, 'train_data': None, 'test_data': None, 'num_clients':0, 'class_path': 'benchmark.domain_digit_classification.core','trans':'transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])'},
        }
        self.domain_balance = (self.dist_id==0)
        self.num_domains = len(self.domain_names)
        clients_per_domain = self.num_clients // self.num_domains
        domain_clients = [clients_per_domain for _ in range(self.num_domains)]
        rest = self.num_clients % self.num_domains
        for i in range(0, rest): domain_clients[i] += 1
        for i in range(self.num_domains):
            self.domain_info[self.domain_names[i]]['num_clients'] = domain_clients[i]
        self.datasrc = [self.create_datasrc(name, name, self.domain_info[name]['class_path'], self.domain_info[name]['trans']) for name in self.domain_names]
        self.save_data = self.multi_save_data

    def load_data(self):
        for domain_name, dinfo in self.domain_info.items():
            try:
                print('Loading Dataset {}'.format(domain_name))
                DomainDataset = self.domain_info[domain_name]['class']
                train_data = DomainDataset(root='/'.join([self.rawdata_path, domain_name]), train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
                test_data = DomainDataset(root='/'.join([self.rawdata_path, domain_name]), train=False, download=True,transform=transforms.Compose([transforms.ToTensor()]))
                self.domain_info[domain_name]['train_data'] = train_data
                self.domain_info[domain_name]['test_data'] = test_data
                self.domain_info[domain_name]['train_size'] = len(train_data)
                self.domain_info[domain_name]['test_size'] = len(test_data)
                print('{} Successfully Loaded'.format(domain_name))
            except:
                print('Failed to download {}. '.format(domain_name))
                exit(1)

    def partition(self):
        if self.domain_balance:
            min_datavol = min([self.domain_info[dataset]['train_size'] for dataset in self.domain_info.keys()])
            local_datas = []
            for domain_id in range(len(self.domain_names)):
                dataset_name = self.domain_names[domain_id]
                d_idxs = np.random.permutation(self.domain_info[dataset_name]['train_size'])
                d_idxs = d_idxs[: min_datavol]
                c_idxs = np.array_split(d_idxs, self.domain_info[self.domain_names[domain_id]]['num_clients'])
                c_idxs = [cidx.tolist() for cidx in c_idxs]
                c_idxs = [[(domain_id, did) for did in cidx] for cidx in c_idxs]
                local_datas.extend(c_idxs)
            return local_datas

    def local_holdout(self, local_datas, rate=0.8, shuffle=False):
        """split each local dataset into train data and valid data according the rate."""
        train_cidxs = []
        valid_cidxs = []
        for local_data in local_datas:
            if shuffle:
                np.random.shuffle(local_data)
            k = int(len(local_data) * rate)
            train_cidxs.append(local_data[:k])
            valid_cidxs.append(local_data[k:])
        return train_cidxs, valid_cidxs

    def multi_save_data(self, train_cidxs, valid_cidxs):
        if self.datasrc ==None:
            raise RuntimeError("Attr datasrc not Found. Please define it in __init__() before calling IndexData_to_json")
        feddata = {
            'store': 'MultiSourceIDX',
            'client_names': self.cnames,
            'dtest': [[domain_id, didx] for domain_id in range(self.num_domains) for didx in range(self.domain_info[self.domain_names[domain_id]]['test_size'])],
            'datasrc': self.datasrc
        }
        for cid in range(self.num_clients):
            feddata[self.cnames[cid]] = {
                'dtrain': train_cidxs[cid],
                'dvalid': valid_cidxs[cid]
            }
        with open(os.path.join(self.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return

    def create_datasrc(self, name, class_name, class_path, trans):
        return {
            'class_path': class_path,
            'class_name': class_name,
            'train_args':{
                'root': "'"+self.rawdata_path+'/'+name+"'",
                'train':'True',
                'download':'True',
                'transform': trans,
            },
            'test_args': {
                'root': "'" + self.rawdata_path + '/' + name + "'",
                'train': 'False',
                'download': 'True',
                'transform': trans,
            }
        }

class TaskPipe(BasicTaskPipe):
    def load_task(self):
        with open(os.path.join(self.taskpath, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        datasrcs = feddata['datasrc']
        for src in datasrcs:
            class_path = src['class_path']
            class_name = src['class_name']
            origin_class = getattr(importlib.import_module(class_path), class_name)
            MultiSourceIDXDataset.ADD_ORIGIN_CLASS(origin_class)
            original_train_data = self.args_to_dataset(src['train_args'])
            original_test_data = self.args_to_dataset(src['test_args'])
            MultiSourceIDXDataset.ADD_NAME(class_name)
            MultiSourceIDXDataset.ADD_SOURCE(original_train_data, original_test_data)
            test_data = MultiSourceIDXDataset(feddata['dtest'], key='TEST')
            train_datas = [MultiSourceIDXDataset(feddata[name]['dtrain']) for name in feddata['client_names']]
            valid_datas = [MultiSourceIDXDataset(feddata[name]['dvalid']) for name in feddata['client_names']]
        return train_datas, valid_datas, test_data, feddata['client_names']

    def args_to_dataset(self, args):
        if not isinstance(args, dict):
            raise TypeError
        args_str = '(' +  ','.join([key+'='+value for key,value in args.items()]) + ')'
        return eval("MultiSourceIDXDataset._DOMAIN_DATASET['CLASS'][-1]"+args_str)

class TaskCalculator(ClassificationCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)

    def test(self, model, dataset, batch_size=64):
        pass

class MNISTM(VisionDataset):
    """MNIST-M Dataset.
    """

    resources = [
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz',
         '191ed53db9933bd85cc9700558847391'),
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz',
         'e11cb4d7fff76d7ec588b1134907db59')
    ]

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        print(os.path.join(self.processed_folder, data_file))

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        """Download the MNIST-M data."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

class SyntheticDigits(VisionDataset):
    """Synthetic Digits Dataset.
    """

    resources = [
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_train.pt.gz',
         'd0e99daf379597e57448a89fc37ae5cf'),
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_test.pt.gz',
         '669d94c04d1c91552103e9aded0ee625')
    ]

    training_file = "synth_train.pt"
    test_file = "synth_test.pt"
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """Init Synthetic Digits dataset."""
        super(SyntheticDigits, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        print(os.path.join(self.processed_folder, data_file))

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        """Download the Synthetic Digits data."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

class SVHN(datasets.SVHN):
    def __init__(self, root, train=True, download = True, transform = None):
        target_transform = transforms.Lambda(lambda y: 0 if y==10 else y)
        if train==True:
            super(SVHN, self).__init__(root, split='train', download=download, transform=transform, target_transform=target_transform)
        else:
            super(SVHN, self).__init__(root, split='test', download=download, transform=transform, target_transform=target_transform)

class MultiSourceIDXDataset(Dataset):
    _DOMAIN_DATASET = {
        'TRAIN':[],
        'TEST':[],
        'CLASS':[],
        'NAME':[],
    }

    def __init__(self, idxs, key='TRAIN'):
        self.idxs = idxs
        self.key = key
        self.domain_ids = set([item[0] for item in idxs])

    def __getitem__(self, item):
        source, idx = self.idxs[item][0], self.idxs[item][1]
        return self._DOMAIN_DATASET[self.key][source][idx]

    def __len__(self):
        return len(self.idxs)

    @classmethod
    def ADD_SOURCE(cls, train_data=None, test_data=None):
        cls._DOMAIN_DATASET['TRAIN'].append(train_data)
        cls._DOMAIN_DATASET['TEST'].append(test_data)

    @classmethod
    def ADD_ORIGIN_CLASS(cls, DataClass = None):
        cls._DOMAIN_DATASET['CLASS'].append(DataClass)

    @classmethod
    def ADD_NAME(cls, name=None):
        if name:
            cls._DOMAIN_DATASET['NAME'].append(name)