from matplotlib import pyplot as plt
import numpy as np


def plot_learning_curve(training_res: np.ndarray, validation_res: np.ndarray, metric: str, title: str, filename: str):
    '''
    plots the learning curve
    
    Parameters
    ----------
    training_res : array_like (1d)
        the training points to plot
    validation_res : array_like (1d)
        the validation points to plot
    metric : str
        the metric that is being plotted 
    title : str
        the title of the plot
    filename : str
        the file to save the plot
    '''
    fig, ax = plt.subplots()

    x = range(training_res.shape[0])

    ax.plot(x, training_res, label=f"Training {metric}")
    ax.plot(x, validation_res, label=f"Testing {metric}")

    ax.legend()
    ax.set(xlabel="Epoch", ylabel=metric)
    ax.set_title(title)

    fig.savefig(filename)
    
def show_img(img, ax, title):
    ax.imshow((img * 255).astype(dtype="uint8").reshape(3, 32, 32).transpose(1, 2, 0))
    if title:
        ax.set(title=title)
    ax.axis("off")


def show_images(images_dict, reconstructed_images_dict, title=None, subtitles=None, filename="default.png"):
    fig, axs = plt.subplots(len(images_dict.keys()), 2)
    if subtitles is None:
        subtitles = [None, None]
    for i, k in enumerate(images_dict):
        show_img(images_dict[k], axs[i][0], title=f"{k} {subtitles[0]}")
        show_img(reconstructed_images_dict[k], axs[i][1], title=f"{k} {subtitles[1]}")
    if title:
        fig.subplots_adjust(top=0.82)
        fig.suptitle(title, ha="center")
    fig.savefig(filename)