import torch
from torch.utils.data import DataLoader
from plots import plot_learning_curve, show_images

class NeuralNetwork():
    """
    Generic neural network module
    """
    def __init__(self, model, optimizer, criterion, device, tag: str):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.tag = tag


    def train_single_epoch(self, dataloader, train_loss_epochs, i):
        '''
        implements the training of the model for a single epoch
        Parameters
        ----------
        dataloader : DataLoader
            a DataLoader for the training data
        train_loss_epochs : Tensor
            a Tensor containing the training loss for each epoch (initialized at 0 for each epoch)
        i : int
            current epoch index
        '''
        samples = torch.tensor(0, dtype=torch.int32, requires_grad=False, device=self.device)
        self.model.train()
        
        for xb, yb in dataloader:
            xb = xb.to(self.device)
            x_hat = self.model(xb)
            loss = self.criterion(x_hat, xb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():
                train_loss_epochs[i] += loss * xb.size(dim=0)
                samples += xb.size(dim=0)
        
        train_loss_epochs[i] /= samples


    def evaluate(self, dataloader, val_loss_epochs, i):
        '''
        evaluates the model
        Parameters
        ----------
        dataloader : DataLoader
            a DataLoader for the evaluating data
        '''
        self.model.eval()
        
        with torch.no_grad():
            samples = torch.tensor(0, dtype=torch.int32, requires_grad=False, device=self.device)
            for xb, yb in dataloader:
                xb = xb.to(self.device)
                x_hat = self.model(xb)
                loss = self.criterion(x_hat, xb)
                
                val_loss_epochs[i] += loss * xb.size(dim=0)
                samples += xb.size(0)
        
            val_loss_epochs[i] /= samples


    def fit(self, dataset, epochs, batch_size=None, images_to_reconstruct=None):
        '''
        implements the training loop
        Parameters
        ----------
        train_dataloader : DataLoader
            a DataLoader for the training data
        val_dataloader : DataLoader
            a DataLoader for the validation data
        epochs: int
            the number of epochs to train the model
            
        Returns
        -------
        dict
            a dict with training and validation losses for every epoch (dict['train'], dict['validation']).
            
        '''
        train_loss = torch.zeros(epochs, dtype=torch.float32, requires_grad=False, device=self.device)
        test_loss = torch.zeros(epochs, dtype=torch.float32, requires_grad=False, device=self.device)

        if batch_size is None:
            batch_size = dataset["train"].shape[0]
        train_dataloader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(dataset["test"], batch_size=min(batch_size, len(dataset["test"])), shuffle=False)
        
        print(f"\rModel: {self.tag} began training", end="")
        for epoch in range(epochs):
            self.train_single_epoch(train_dataloader, train_loss, epoch)
            self.evaluate(val_dataloader, test_loss, epoch)
            print(f"\rModel: {self.tag} Epoch: {epoch + 1} Training loss: {train_loss[epoch]} Testing loss: {test_loss[epoch]}", end="")
        print()
        
        train_loss = train_loss.cpu().numpy()
        test_loss = test_loss.cpu().numpy()
        
        # Plot learning curves                     
        plot_learning_curve(training_res=train_loss, validation_res=test_loss, metric='MSE', title=self.tag, filename=f'{self.tag}_loss.png')

        # Show the chosen image from the dataset
        if images_to_reconstruct is not None:
              reconstructed_images = {k: self.model(torch.tensor(images_to_reconstruct[k][None, :], dtype=torch.float32, requires_grad=False, device=self.device)).detach().cpu().numpy() for k in images_to_reconstruct}
              show_images(images_to_reconstruct, 
                          reconstructed_images, 
                          title=None, 
                          subtitles=["original", "reconstructed"])
            
        return {'train': train_loss, 'test': test_loss}