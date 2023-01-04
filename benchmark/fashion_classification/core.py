from torchvision import datasets, transforms
from benchmark.toolkits import DefaultTaskGen
from benchmark.toolkits import ClassificationCalculator as TaskCalculator
from benchmark.toolkits import IDXTaskPipe as TaskPipe
class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, seed=0):
        super(TaskGen, self).__init__(benchmark='fashion_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/FASHION',
                                      seed=seed
                                      )
        self.num_classes = 10
        self.label_dict = {0: 'T-shirt', 1: 'Trouser', 2: 'pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Abkle boot'}
        self.save_task = TaskPipe.save_task
        self.visualize = self.visualize_by_class
        self.source_dict = {
            'class_path': 'torchvision.datasets',
            'class_name': 'FashionMNIST',
            'train_args': {
                'root': '"' + self.rawdata_path + '"',
                'download': 'True',
                'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])',
                'train': 'True'
            },
            'test_args': {
                'root': '"' + self.rawdata_path + '"',
                'download': 'True',
                'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])',
                'train': 'False'
            }
        }

    def load_data(self):
        self.train_data = datasets.FashionMNIST(self.rawdata_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_data = datasets.FashionMNIST(self.rawdata_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

