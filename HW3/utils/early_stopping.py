

class EarlyStopping:
    """
    Stops training when validation loss stops improving.
    """

    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.stop = False

    def step(self, val_loss):

        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop = True
