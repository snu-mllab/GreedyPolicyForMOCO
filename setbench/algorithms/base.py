import wandb
import pickle
import gzip
import os

class BaseAlgorithm():
    def __init__(self, cfg, task, tokenizer, task_cfg, **kwargs):
        self.cfg = cfg
        self.task = task
        self.tokenizer = tokenizer
        self.task_cfg = task_cfg
        self.state = {}

    def optimize(self, task, initial_data=None):
        raise NotImplementedError("Override this method in your class")
    
    def log(self, metrics, commit=True):
        wandb.log(metrics, commit=True)
    
    def update_state(self, metrics):
        for k, v in metrics.items():
            if k in self.state.keys():
                self.state[k].append(v)
            else:
                self.state[k] = [v]

    def save_state(self):
        with gzip.open(self.state_save_path, 'wb+') as f:
            pickle.dump(self.state, f)
