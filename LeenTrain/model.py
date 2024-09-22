"""

Define your model here using the boilerplate code below, and make sure to include the `forward` method.

"""
import torch
import torch.nn as nn
from loguru import logger
from torchsummary import summary

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
        self.convolutions = nn.Sequential(
            nn.Conv2d(self.in_channels, self.filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filters),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2),  # Output size halved
            nn.Conv2d(self.filters, self.filters*2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.filters*2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2),  # Output size halved again
            nn.Conv2d(self.filters*2, self.filters*3, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.filters*3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2),
            # Output size halved once more
        )

        # Calculate the flattened size based on actual output shape after convolutions
        flattened_size = self._get_flattened_size(input_size)
        logger.info(f"Flattened size for the first Linear layer: {flattened_size}")

        # Remove AdaptiveAvgPool2d, as the tensor is already reduced
        self.dense = nn.Sequential(
            nn.Flatten(),  # Flatten the 2D to 1D
            nn.Linear(flattened_size, self.units1),  # Input size should match the flattened size
            nn.BatchNorm1d(self.units1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.units1, self.units2),
            nn.BatchNorm1d(self.units2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.units2, self.units2),
            nn.BatchNorm1d(self.units2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
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