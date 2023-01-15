from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

def load_data(params):
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_train = CIFAR10(root = './',
                    train = True,
                    download = False,
                    transform = trans)
    trainloader = DataLoader(data_train,
                            batch_size = params.batchsize,
                            shuffle = True,
                            num_workers = params.numworkers,
                            drop_last = True,
                            pin_memory = True)
    return trainloader
def transback(data):
    return data / 2 + 0.5