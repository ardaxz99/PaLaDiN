from header import *
from .distributed_samplers import DistributedBatchSampler
from .mvtec_dataset import *
from .visa_dataset import *

def load_mvtec_dataset(args):

    # data = MVtecDataset(args['dataset_path'])
    data = MVtecDatasetGuided(args['dataset_path'])

    if not os.path.exists(args['dataset_path']):
        raise FileNotFoundError(f"Dataset path does not exist: {args['dataset_path']}")

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=8,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler


def load_visa_dataset(args):

    # data = VisaDataset(args['dataset_path'])
    data = VisaDatasetGuided(args['dataset_path'])

    if not os.path.exists(args['dataset_path']):
        raise FileNotFoundError(f"Dataset path does not exist: {args['dataset_path']}")
    
    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=8,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler
