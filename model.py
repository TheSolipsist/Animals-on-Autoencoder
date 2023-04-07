import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self, conv_channels=16, kernel_size=3, encoding_dim=1024, stride=2, sampling_scale=None, image_dim=256):
        '''
        Autoencoder model

        Parameters
        ----------
        conv_channels : int
            the number of output channels for the convolutional encoder (input for decoder)
        kernel_size : int
            kernel size for the convolutional layer
        encoding_dim : int
            the number of output neurons in the final linear layer of the encoding module (bottleneck layer size)
        stride : int
            kernel's stride for the convolutional layer
        sampling_scale : int
            size of kernel window for MaxPool2d and scale for UpsamplingNearest2d (no downsampling/upsampling if None)
        image_dim : int
            the size of the RGB square input image
        '''
        super().__init__()
        def calc_dim(dim_len, kernel_size, padding, stride, sample_kernel_size):
            if sample_kernel_size is None:
                sample_kernel_size = 1
            return ((dim_len - kernel_size + 2 * padding + stride) // stride) // sample_kernel_size
        
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = self.kernel_size // 2
        self.sampling_scale = sampling_scale
        self.image_dim = image_dim
        self.encoding_dim = encoding_dim
        
        self.dimension_conv1 = calc_dim(image_dim, self.kernel_size, self.padding, self.stride, self.sampling_scale)
        # # self.dimension_conv2 = calc_dim(self.dimension_conv1, self.kernel_size, self.padding, self.stride, 1)
        self.dimension_fc1 = self.conv_channels *  (self.dimension_conv1 ** 2)
        
        self.encoder_cnn1 = nn.Conv2d(3, self.conv_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.encoder_cnn_act1 = nn.ReLU()
        # self.encoder_cnn2 = nn.Conv2d(self.conv_channels[0], self.conv_channels[1], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # self.encoder_cnn_act2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.encoder_fc1 = nn.Linear(self.dimension_fc1, encoding_dim)
        self.encoder_act1 = nn.ReLU()
        # self.encoder_fc2 = nn.Linear(self.linear_layer_dims[1], self.linear_lay``er_dims[2])
        # self.encoder_act2 = nn.ReLU()
        self.decoder_fc1 = nn.Linear(encoding_dim, self.dimension_fc1)
        self.decoder_act1 = nn.ReLU()
        # self.decoder_fc2 = nn.Linear(self.linear_layer_dims[-2], self.linear_layer_dims[-3])
        # self.decoder_act2 = nn.ReLU()
        self.unflatten = nn.Unflatten(1, (self.conv_channels, self.dimension_conv1, self.dimension_conv1))
        # self.decoder_upsample = nn.UpsamplingNearest2d(scale_factor=self.sampling_scale)
        self.decoder_cnn1 = nn.ConvTranspose2d(self.conv_channels, 3, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=3)
        self.decoder_cnn_act1 = nn.Sigmoid()
        # self.decoder_cnn2 = nn.ConvTranspose2d(self.conv_channels[-1], self.conv_channels[-2], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=1)
        # self.decoder_cnn_act2 = nn.Sigmoid()
        if sampling_scale:
            self.encoder_downsample = nn.MaxPool2d(kernel_size=self.sampling_scale)
            self.decoder_upsample = nn.UpsamplingNearest2d(scale_factor=self.sampling_scale)
            self.encoder_cnn = nn.Sequential(
                self.encoder_cnn1,
                self.encoder_cnn_act1,
                self.encoder_downsample
            )
            self.decoder_cnn = nn.Sequential(
                self.decoder_upsample,
                self.decoder_cnn1,
                self.decoder_cnn_act1
            )
        else:
            self.encoder_cnn = nn.Sequential(
                self.encoder_cnn1,
                self.encoder_cnn_act1
            )
            self.decoder_cnn = nn.Sequential(
                self.decoder_cnn1,
                self.decoder_cnn_act1
            )

        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.encoder_act1(self.encoder_fc1(self.flatten(x)))
        x = self.unflatten(self.decoder_act1(self.decoder_fc1(x)))
        x = self.decoder_cnn(x)

        return x
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unflatten(self.decoder_act1(self.decoder_fc1(x)))
        x = self.decoder_cnn(x)
        
        return x
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_cnn(x)
        x = self.encoder_act1(self.encoder_fc1(self.flatten(x)))
        
        return x