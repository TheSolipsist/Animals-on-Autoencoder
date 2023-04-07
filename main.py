import torch
from preprocessing import load_images, train_test_split, get_imagefolder
from neural_net import NeuralNetwork
from model import Autoencoder
import os
from generate_samples import generate_sample, reconstruct_image


def create_model(image_dim=128, encoding_dim=256):
    datasets = train_test_split(load_images(size=image_dim, device=torch.device("cuda")), train_size=0.8)

    model = Autoencoder(conv_channels=16,
                        kernel_size=5,
                        stride=4,
                        encoding_dim=encoding_dim,
                        image_dim=image_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    nn = NeuralNetwork(model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    tag="Autoencoder", 
                    device=torch.device("cuda"))
    path = f"saved_models/{image_dim}_{encoding_dim}"
    if not os.path.exists(path):
        os.mkdir(path)
    nn.fit(datasets, epochs=200, batch_size=64, plot_path=path)
    torch.save(nn.model, f"{path}/model.pt")
    
def generate_samples(n_images, image_dim=128, encoding_dim=256):
    import random
    model = torch.load(f"saved_models/{image_dim}_{encoding_dim}/model.pt").to("cpu")
    folder = get_imagefolder(size=image_dim)
    while True:
        images = [folder[random.randint(0, len(folder) - 1)][0].reshape(1, 3, image_dim, image_dim) for _ in range(n_images)]
        generate_sample(model=model, 
                        images=images)

def reconstruct_images(image_dim=128, encoding_dim=1024):
    model = torch.load(f"saved_models/{image_dim}_{encoding_dim}/model.pt").to("cpu")
    folder = get_imagefolder(size=image_dim)
    import random
    while True:
        idx = random.randint(0, len(folder) - 1)
        reconstruct_image(folder, idx, model, image_dim)

generate_samples(n_images=2, image_dim=64, encoding_dim=1024)