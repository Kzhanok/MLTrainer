import torch
import torch.nn as nn
from loguru import logger

class ModelBlocks(nn.Module):
    def __init__(self):
        super().__init__()

    def linear_block(self, in_features, out_features, batch_norm=False, dropout=0.0):
        layers = [nn.Linear(in_features, out_features)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

    def conv2d_max_block(self, in_channels, out_channels, kernel_size, stride, padding,
                         batch_norm=False, max_pool=False, max_pool_kernel_size=2):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        if max_pool:
            print(max_pool_kernel_size)
            layers.append(nn.MaxPool2d(max_pool_kernel_size))
        return nn.Sequential(*layers)
    
    def conv_transition_block(self, output_size: int):
    # Adaptive pooling to adjust spatial dimensions to (output_size, output_size)
    # Followed by a flattening operation to prepare for linear layers
        return nn.Sequential(
        nn.AdaptiveAvgPool2d((output_size, output_size)),
        nn.Flatten()  # Flatten the (batch_size, channels, output_size, output_size) to (batch_size, channels * output_size * output_size)
    )

class Model(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.to(self.device)

        if config is None:
            logger.error("No configuration provided.")
            return

        self.input_size = config['model']['input_size']
        
        self.blocks = ModelBlocks()
        self.layers = nn.ModuleList()
        self._build_model(config['model']['layers'])
        self.to(self.device)

    def _build_model(self, layer_configs):
        for layer_cfg in layer_configs:
            layer_type = layer_cfg['type']
            params = layer_cfg['params']
            layer = getattr(self.blocks, layer_type)(**params)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def output_model_summary(self):
        from torchsummary import summary
        summary(self, tuple(self.input_size), device=str(self.device))

if __name__ == "__main__":
    import yaml
    with open('LeenTrain\config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model = Model(config)
    model.output_model_summary()
