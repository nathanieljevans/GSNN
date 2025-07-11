class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        """
        Early stopping implementation for neural network training.

        Tracks validation loss and stops training when no improvement is seen for a specified number of epochs.

        Original source: @isle_of_gods (https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch)

        Args:
            patience (int, optional): Number of epochs to wait for improvement before stopping. Default: 1
            min_delta (float, optional): Minimum change in validation loss to qualify as an improvement. Default: 0

        Example:
            >>> early_stopper = EarlyStopper(patience=5, min_delta=0.001)
            >>> for epoch in range(100):
            ...     val_loss = train_epoch()
            ...     if early_stopper.early_stop(val_loss):
            ...         print(f'Stopping early at epoch {epoch}')
            ...         break
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        else: 
            self.counter += 1
            if self.counter > self.patience:
                return True
            
        return False