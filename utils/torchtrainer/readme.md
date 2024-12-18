# PyTorch Training Framework

This repository provides a versatile PyTorch training framework to simplify and enhance the model training process. It includes support for metrics tracking, early stopping, and customizable callbacks.

## Features

- **Metrics Tracking:** Calculate and monitor multi-class and binary accuracy, precision, recall, and R² score.
- **Custom Callbacks:** Implement and use custom callbacks for various training events.
- **Early Stopping:** Automatically halt training based on validation loss to avoid overfitting.
- **Mixed Precision Training:** Utilize mixed precision for improved performance on CUDA-enabled GPUs.
- **Detailed Reporting:** Get clear and comprehensive reports of training and validation metrics.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/paraglondhe098/torchtrainer.git
    cd torchtrainer
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Here is a basic example of how to use the `Trainer` class:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtrainer import Trainer, IntraEpochReport, EarlyStopping

# Define model, criterion, and optimizer
model = nn.Sequential(nn.Linear(10, 1))
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    epochs=10,
    criterion=criterion,
    optimizer=optimizer,
    metrics=['accuracy'],
    mixed_precision_training=True
)

# Add callbacks
trainer.add_callback(IntraEpochReport(reports_per_epoch=10))
trainer.add_callback(EarlyStopping(basis='vloss', patience=3))

# Prepare your data loaders
train_loader = ...
val_loader = ...

# Train the model
trainer.train(train_loader, val_loader)
```


