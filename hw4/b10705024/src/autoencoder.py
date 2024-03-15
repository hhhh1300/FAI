import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

"""
Implementation of Autoencoder
"""
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class customDataset(Dataset):
    def __init__(self, X):
        self.tensors = X
        
    def __getitem__(self, index):
        x = self.tensors[index]
        return x
        
    def __len__(self):
        return len(self.tensors)

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim//2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.Tanh()
        )
        self.training_curve = list()
    
    def forward(self, x):
        #TODO: 5%
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def fit(self, X, epochs=10, batch_size=32, learning_rate=1e-4, opti=torch.optim.AdamW):
        #TODO: 5%
        dataset_X = customDataset(X)
        dataloader_X = DataLoader(dataset_X, batch_size=batch_size)
        criterion = nn.MSELoss()
        optimizer = opti(self.parameters(), lr=learning_rate)
        self.train()
        for epoch in range(epochs):
            total_loss = list()
            for data in dataloader_X:
                data = data.to(torch.float32)
                loss = criterion(self(data), data)
                total_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.training_curve.append(np.mean(total_loss))
            # print(f'{epoch} : avg loss {np.mean(total_loss)}')
    
    def transform(self, X):
        #TODO: 2%
        X = torch.tensor(X).to(torch.float32)
        trans_X = self.encoder(X)
        return trans_X.detach().numpy()
    
    def reconstruct(self, X):
        #TODO: 2%
        X = torch.tensor(X).to(torch.float32)
        re_X = self(X)
        return re_X.detach().numpy()
    


"""
Implementation of DenoisingAutoencoder
"""
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim,encoding_dim)
        self.noise_factor = noise_factor
    
    def add_noise(self, x):
        #TODO: 3%
        return x + torch.normal(mean=0.0, std=self.noise_factor, size=(1,)).item()
    
    def fit(self, X, epochs=10, batch_size=32, learning_rate=1e-4, opti=torch.optim.AdamW):
        #TODO: 4%
        dataset_X = customDataset(X)
        dataloader_X = DataLoader(dataset_X, batch_size=batch_size)
        criterion = nn.MSELoss()
        optimizer = opti(self.parameters(), lr=learning_rate)
        self.train()
        for epoch in range(epochs):
            total_loss = list()
            for data in dataloader_X:
                noised_data = torch.zeros_like(data)
                for i, x in enumerate(data):
                    noised_data[i] = self.add_noise(x)
                data = data.to(torch.float32)
                noised_data = noised_data.to(torch.float32)
                loss = criterion(self(noised_data), data)
                total_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.training_curve.append(np.mean(total_loss))
            # print(f'{epoch} : avg loss {np.mean(total_loss)}')
