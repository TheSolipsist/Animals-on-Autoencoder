import torch
from preprocessing import load_imagefolder, train_test_split
from matplotlib import pyplot as plt
from neural_net import NeuralNetwork
from model import Autoencoder

datasets = train_test_split(load_imagefolder(), train_size=0.8)
model = Autoencoder(conv_channels=16,
                    kernel_size=9, 
                    stride=4,
                    encoding_dim=1024)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
nn = NeuralNetwork(model=model,
                   optimizer=optimizer,
                   criterion=criterion,
                   tag="Autoencoder", 
                   device=torch.device("cuda"))
nn.fit(datasets, epochs=200, batch_size=64)