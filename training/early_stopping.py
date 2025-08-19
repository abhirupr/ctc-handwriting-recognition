import numpy as np
import torch
import os

class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(self, patience=7, min_delta=0.0001, restore_best_weights=True, 
                 mode='max', baseline=None, verbose=True):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best value.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the quantity 
                       has stopped decreasing; in 'max' mode it will stop when the quantity has stopped increasing.
            baseline (float): Baseline value for the monitored quantity. Training will stop if the model doesn't 
                            show improvement over the baseline.
            verbose (bool): Whether to print early stopping messages.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.baseline = baseline
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            raise ValueError(f"Mode {mode} is unknown, please use 'min' or 'max'.")
            
        if self.baseline is not None:
            self.best = self.baseline
    
    def __call__(self, current, model=None):
        """
        Call this method after each epoch.
        
        Args:
            current (float): Current value of the monitored metric
            model (torch.nn.Module): Model to save weights from
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights and model is not None:
                # Deep copy tensors to avoid in-place mutation during further training steps
                self.best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"🎯 New best metric: {current:.4f}")
        else:
            self.wait += 1
            if self.verbose and self.wait == 1:
                print(f"⏳ No improvement for 1 epoch (best: {self.best:.4f})")
            elif self.verbose and self.wait > 1:
                print(f"⏳ No improvement for {self.wait} epochs (best: {self.best:.4f})")
                
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                if self.verbose:
                    print(f"🛑 Early stopping triggered after {self.wait} epochs without improvement!")
                    print(f"   Best metric was {self.best:.4f}")
                return True
        
        return False
    
    def restore_weights(self, model):
        """Restore the best weights to the model."""
        if self.best_weights is not None and model is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f"✅ Restored best weights (metric: {self.best:.4f})")
        else:
            if self.verbose:
                print("⚠️ No best weights to restore")
    
    def get_best_metric(self):
        """Get the best metric value seen so far."""
        return self.best
    
    def reset(self):
        """Reset the early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        if self.mode == 'min':
            self.best = np.Inf
        else:
            self.best = -np.Inf
        if self.baseline is not None:
            self.best = self.baseline