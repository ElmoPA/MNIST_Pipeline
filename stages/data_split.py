import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torchvision .transforms import ToTensor

def calc_mean_std(dataset, batch_size=64, shuffle=False):
    mean = 0.0
    std = 0.0
    length = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    for images, _ in loader:
        mean += images.mean()
        std += images.std()
        length += 1
    mean /= length
    std /= length
    print(f'mean: {mean}\nstd: {std}')
    return mean, std


if __name__ == '__main__':
    import yaml
    import argparse

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config')
    args = args_parser.parse_args()

    with open(args.config, 'r') as f:
        param = yaml.safe_load(f)   
    base = param['base']
    data_dir = base['data_dir']

    calc_set = datasets.MNIST(
        root=data_dir+'/train',
        train=True,
        download=False,
        transform=ToTensor()
    )
    mean, std = calc_mean_std(calc_set)
    
    transform = v2.Compose([
        ToTensor(),
        v2.Normalize(mean=[mean.item()], std=[std.item()])
    ])

    train_set = datasets.MNIST(
        root=data_dir+'/train',
        train=True,
        download=False,
        transform=None
    )
    test_set = datasets.MNIST(
        root=data_dir+'/test',
        train=False,
        download=False,
        transform=None
    )

    train_image = [transform(image) for image, label in train_set]
    train_image = torch.stack(train_image, dim=0)
    train_label = [torch.tensor(label) for image, label in train_set]
    train_label = torch.stack(train_label, dim=0)

    test_image = [transform(image) for image, label in test_set]
    test_image = torch.stack(test_image, dim=0)
    test_label = [torch.tensor(label) for image, label in test_set]
    test_label = torch.stack(test_label, dim=0)

    torch.save(train_image, data_dir+'/train_image.pt')
    torch.save(train_label, data_dir+'/train_label.pt')
    torch.save(test_image, data_dir+'/test_image.pt')
    torch.save(test_label, data_dir+'/test_label.pt')








