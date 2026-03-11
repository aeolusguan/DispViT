from typing import Callable, Optional
import random
from abc import ABC
import torch

from .worker_fn import get_rank, get_world_size, get_worker_init_fn


class DataLoader(ABC):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        mode: str,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset

        world_size = get_world_size()
        self.batch_size = batch_size // world_size if mode == "train" else 1
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed

        self.num_of_resolutions = getattr(dataset, "num_of_resolutions", 1)

        # Create samplers
        if mode == "train":
            self.sampler = DynamicDistributedSampler(self.dataset, seed=seed, shuffle=shuffle)
            self.batch_sampler = DynamicBatchSampler(
                self.sampler,
                self.num_of_resolutions,
                self.batch_size,
                seed=seed,
            )
        elif mode == "val":
            self.batch_sampler = InferenceSampler(len(self.dataset))
        else:
            raise ValueError(f"Unrecognized mode: {mode}")
        
    def get_loader(self, epoch):
        # Set the epoch for the sampler
        if hasattr(self.batch_sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        # Create and return the dataloader
        return torch.utils.data.DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=self.num_workers,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
            drop_last=self.drop_last,
            prefetch_factor=4,
        )


class InferenceSampler(torch.utils.data.Sampler):
    """
    Produces indices for inference across all workers.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size):
        """
        Args:
            size (int): the total number of data on the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = get_rank()
        self._world_size = get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_size = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_size[:rank])
        end = min(sum(shard_size[: rank + 1]), total_size)
        return range(begin, end)
    
    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
    

class DynamicBatchSampler(torch.utils.data.Sampler):
    """"
    A custom batch sampler that dynamically adjusts resolution for each sample.
    Batches within a sample share the same resolution.
    """
    def __init__(self,
                 sampler,
                 pool_size,
                 batch_size_per_gpu,
                 epoch=0,
                 seed=42):
        """"
        Initialize the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            pool_size: Number of different resolutions to sample from.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
        """
        self.sampler = sampler
        self.pool_size = pool_size
        self.rng = random.Random()

        # Batch size per GPU
        self.batch_size_per_gpu = batch_size_per_gpu

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):
        """
        Yields batches of samples with synchronized dynamic parameters.

        Returns:
            Iterator yielding batches of indices and associated parameters.
        """
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                # Sample a random resolution index for current batch
                random_resolution_index = self.rng.randint(0, self.pool_size - 1)
                self.sampler.update_parameters(random_resolution_index)

                # Collect samples for the current batch
                current_batch = []
                for _ in range(self.batch_size_per_gpu):
                    try:
                        item = next(sampler_iterator)  # item is (idx, crop_size_index)
                        current_batch.append(item)
                    except StopIteration:
                        break  # No more samples
                if not current_batch:
                    break  # No more data to yield

                yield current_batch

            except StopIteration:
                break  # End of sampler's iterator


class DynamicDistributedSampler(torch.utils.data.DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic resolution parameters,
    which can be passed into the dataset's __getitem__ method.
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        self.crop_size_index = None
    
    def __iter__(self):
        """
        Yields a sequence of (index, crop_size_index).
        Relies on the parent class's logic for shuffling and distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for idx in indices_iter:
            yield (idx, self.crop_size_index,)

    def update_parameters(self, crop_size_index):
        """
        Update dynamic parameters for each new epoch or iteration.
        
        Args:
            crop_size_index (int): An index representing the crop size to be used in the dataset's __getitem__ method.
        """
        self.crop_size_index = crop_size_index