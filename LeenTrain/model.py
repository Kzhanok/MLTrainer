"""

Define your model here using the boilerplate code below, and make sure to include the `forward` method.

"""
import torch
import torch.nn as nn
from loguru import logger
from torchsummary import summary


"""
Define model blocks here

"""
class model_blocks(nn.Module):
    def __init__(self):
        super().__init__()
        
    """
    Define your model blocks here
    """    
    def linear_block(self, in_features: int, out_features: int, batch_norm: bool = False, dropout: float = 0.0):
        layers = []
        layers.append(nn.Linear(in_features, out_features))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU())
        if dropout==0.0:
            layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)
    
    def conv2d_max_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, batch_norm: bool = False, max_pool: bool = False):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        return nn.Sequential(*layers)
    
    def res_linear_block(self, in_features: int, out_features: int, batch_norm: bool = False, dropout: float = 0.0):
        layers = []
        layers.append(nn.Linear(in_features, out_features))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        
        # Define the residual connection
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
        return nn.Sequential(*layers)
    
    
class Model(nn.Module):
    def __init__(self,params: dict = None,input_size: tuple = None,device: str = "cuda:0",):
        #Sets device to device, default is cuda:0 and inherit nn.Module
        torch.set_default_device(device)
        
    
        
        #sets all parameters as attributes
        for key, value in params.items():
            setattr(self, key, value)
        
        #checks if all parameters are provided
        if params == None or input_size == None:
            logger.info("Not all parameters or input size provided.")
            return
        
        self.input_size = input_size
        self.in_channels = input_size[1]
        super().__init__()
        
        """
        Define your model here, parameters are passed as self.paramname
        """    
        #TODO set model to be instantiated with ModelSettings
        # Define the convolutional layers using the conv2d_max_block
        self.convolutions = nn.Sequential(
            model_blocks().conv2d_max_block(self.in_channels, self.filters, kernel_size=3, stride=1, padding=1, batch_norm=True, max_pool=True),
            model_blocks().conv2d_max_block(self.filters, self.filters*2, kernel_size=3, stride=1, padding=0, batch_norm=True, max_pool=True),
            model_blocks().conv2d_max_block(self.filters*2, self.filters*3, kernel_size=3, stride=1, padding=0, batch_norm=True, max_pool=True),
            nn.Dropout(p=0.2)
        )

        # Calculate the flattened size based on actual output shape after convolutions
        flattened_size = self._get_flattened_size(input_size)
        logger.info(f"Flattened size for the first Linear layer: {flattened_size}")

        # Define the dense layers using the linear_block
        self.dense = nn.Sequential(
            nn.Flatten(),  # Flatten the 2D to 1D
            model_blocks().linear_block(flattened_size, self.units1, batch_norm=True, dropout=0.2),
            model_blocks().linear_block(self.units1, self.units2, batch_norm=True, dropout=0.2),
            model_blocks().linear_block(self.units2, self.units2, batch_norm=True, dropout=0.2),
            nn.Linear(self.units2, 10)  # Output 10 classes
        )

    """
    Define your model functions here
    """    
    
    # This function calculates the flattened size after convolutions
    def _get_flattened_size(self, input_size):
        x = torch.ones(1, *input_size[1:], dtype=torch.float32)  # Add batch dimension
        x = self.convolutions(x)
        return x.numel()  # Return the total number of elements (flattened size)

    """
    Define your forward function here
    """    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define the forward pass here
        x = self.convolutions(x)
        x = self.dense(x)  # Forward to dense layers
        return x
    
    """
    model helper functions
    """
    
    def output_model_summary(self):
        # Use the torchsummary.summary function to print the model summary
        tensor_info = self.input_size[1:]
        summary(self, tensor_info)

if __name__ == "__main__":
    model = Model({'filters':128, 'units1':128, 'units2':64},input_size=(32, 1, 28, 28))
    model.output_model_summary()