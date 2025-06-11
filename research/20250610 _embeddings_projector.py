#%%
import numpy
import torch
import os
from tensorboard.plugins import projector
from torch.utils.tensorboard import SummaryWriter
#%%
writer = SummaryWriter('log_dir/')
#%%
def select_n_random():
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    embedding_tensor = torch.tensor([
        [1.0, 0.0, 0.0, 1.0],
    ])

    labels = torch.tensor([0])
    # select random images and their target indices

    # log embeddings
    features = embedding_tensor.view(-1, 4)
    writer.add_embedding(features,
                        metadata=labels.tolist())
select_n_random()