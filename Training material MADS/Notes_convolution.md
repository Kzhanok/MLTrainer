Model:

self.convolutions = nn.Sequential(
            nn.Conv2d(self.in_channels, filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output size halved
            nn.Conv2d(filters, filters*2, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(filters*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output size halved again
            nn.Conv2d(filters*2, filters*3, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(filters*3),
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
            nn.Linear(flattened_size, units1),  # Input size should match the flattened size
            nn.BatchNorm1d(units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.BatchNorm1d(units2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(units2, 5)  # Output 10 classes
        )
Training time rather quick ~1 min, best performance at 73%, but model drops in performance after a few epochs
    - why?
    - Different parameters tested, funnel in dense and less output layers preferable.

model 2 (added two conv layers without reduction):
self.convolutions = nn.Sequential(
            nn.Conv2d(self.in_channels, filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output size halved
            nn.Conv2d(filters, filters*2, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(filters*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output size halved again
            nn.Conv2d(filters*2, filters*3, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(filters*3),
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
            nn.Linear(flattened_size, units1),  # Input size should match the flattened size
            nn.BatchNorm1d(units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.BatchNorm1d(units2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(units2, 5)  # Output 10 classes
        )
notes:
- performance at 64, 512 and 128 is similar to 64 512, 64 both at ~ 72. This model has added two conv layers without reduction which did not decrease performance. 
    So there is a limit to what conv layers can abstract?
- 128 filters slows down training considerable and decreased performance
- removing dropout at 64, 512, 128 did not increase performance. So some is good for generelizability
- what if we increase the units to support the amount of linear units?
    - 64, 1024, 256 still has fast training times and stabilizes alot better - > not repeatable
    - double it both again - did not improve much in terms of stability
    - more epochs caused overfitting of the train set. at 128 filters 1028 256
    - back to 64, 256, 128 -> 74% then it started overfitting -> smaller batches? **Note** still had batchsize at 64 -> reduced to 32 -> did not solve overfitting
    - How do we get the model not to overfit?
