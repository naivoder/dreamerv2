import collections
import datetime
import pathlib
import io 
import uuid

import numpy as np 
import torch

def load_episodes(save_dir, capacity, min_len):
    pass

def count_episodes(save_dir):
    pass

def episode_length(episode):
    pass

class ReplayBuffer:
    def __init__(self, save_dir="logs", capacity=1000, resume=False, min_len=1, max_len=0, prioritize_ends=False):
        self.save_dir = pathlib.Path(save_dir).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.resume = resume
        self.prioritize_ends = prioritize_ends

        self.capacity = capacity
        self.min_len = min_len
        self.max_len = max_len  

        self.completed = load_episodes(self.save_dir, capacity, min_len)
        self.ongoing = collections.defaultdict(lambda: collections.defaultdict(list))

        self.total_episodes, self.total_steps = count_episodes(save_dir)
        self.loaded_episodes = len(self.completed)
        self.loaded_steps = sum(episode_length(x) for x in self.completed.values())
        