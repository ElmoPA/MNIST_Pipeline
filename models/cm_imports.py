from dvclive import Live
import torch
from torch import nn, save, load
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')