import sys
import json
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

from models.BasicCNN import BasicCNN
import torch
from torch import save
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import yaml
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Trainning Loop
def train(train_set, test_set, model, optimizer, epochs, **kwargs):
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    match model:
        case 'BasicCNN':
            clf = BasicCNN()
    clf.to(device)
    match optimizer:
        case 'Adam':
            opt = torch.optim.Adam(clf.parameters(), lr=kwargs['lr'])
    
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = 0.0
        test_loss = 0.0
        count = 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            yhat = clf(X)
            loss = loss_fn(yhat, y)
            train_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                yhat = clf(X)
                loss = loss_fn(yhat, y)
                test_loss += loss.item()
                count+= 1
        print(f'epoch: {epoch}, train_loss: {train_loss/count}, test_loss: {test_loss/count}')
        data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss
        }
        with open('metrics.json', 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', dest='config')
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        param = yaml.safe_load(f)
    base = param['base']

    data_dir = base['data_dir']
    train_image = torch.load(data_dir + '/train_image.pt')
    train_label = torch.load(data_dir + '/train_label.pt')
    test_image = torch.load(data_dir + '/test_image.pt')
    test_label = torch.load(data_dir + '/test_label.pt')
    
    train_set = TensorDataset(train_image, train_label)
    test_set = TensorDataset(test_image, test_label)

    train_param = base['train']
    train(train_set, test_set,
        train_param['model_type'], 
        train_param['optimizer'], 
        train_param['epoch'],
        lr=train_param['lr'])