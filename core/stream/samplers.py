import torch

import numpy as np

class ExtendedSampler(torch.utils.data.Sampler):

    def __init__(self, inds, shuffle=False, repeats=1) -> None:
        self.inds = inds
        self.shuffle = shuffle

        if repeats > 1:
            self.inds = np.repeat(self.inds, repeats)
            
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.inds)

        return iter(self.inds)
        
    def __len__(self) -> int:
        return len(self.inds)
    




