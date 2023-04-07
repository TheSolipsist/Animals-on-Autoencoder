import torch
from matplotlib import pyplot as plt


def generate_sample(model, images):
    with torch.no_grad():
        encs = [model.encode(img) for img in images]
        image_tensor = model.decode(sum(encs) / len(images))
        fig, ax = plt.subplots(len(images) + 1)
        for i, v in enumerate(images):
            ax[i].imshow(v.squeeze().permute(1, 2, 0).detach().numpy())
        ax[-1].imshow(image_tensor.squeeze().permute(1, 2, 0).detach().numpy())
        plt.show()
    
def reconstruct_image(image_folder, idx, model, image_dim=128):
    with torch.no_grad():
        fig, ax = plt.subplots(2)
        ax[0].imshow(image_folder[idx][0].permute(1, 2, 0))
        ax[1].imshow(model(image_folder[idx][0].reshape((1, 3, image_dim, image_dim))).squeeze().permute(1, 2, 0).detach().numpy())
        plt.show()
    