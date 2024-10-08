import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from model import Model
from torchvision import datasets, transforms

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if output.shape != target.shape:
            target = target.view(-1, 1).float()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}: [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item()}')

def main():
    config = load_config('LeenTrain\config.yaml')
    device = torch.device(config['trainer']['device'] if torch.cuda.is_available() else 'cpu')
    model = Model(config, device=device)

    # Define dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['trainer']['learning_rate'])

    # Training loop
    for epoch in range(1, config['trainer']['epochs'] + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)

if __name__ == '__main__':
    main()
