from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data.distributed import DistributedSampler

def load_data(batchsize:int, numworkers:int) -> tuple[DataLoader, DistributedSampler]:
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_train = CIFAR10(
                        root = './',
                        train = True,
                        download = False,
                        transform = trans
                    )
    sampler = DistributedSampler(data_train)
    trainloader = DataLoader(
                        data_train,
                        batch_size = batchsize,
                        num_workers = numworkers,
                        sampler = sampler,
                        drop_last = True
                    )
    return trainloader, sampler

def transback(data:Tensor) -> Tensor:
    return data / 2 + 0.5
