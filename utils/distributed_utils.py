import os
import torch
import torch.distributed as dist

def init_distributed_mode(distributed_config):
    """
    Initialize the distributed training environment.

    Args:
        distributed_config (dict): Configuration dictionary containing distributed parameters.
            - 'backend': Backend to use ('nccl', 'gloo', etc.).
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Initialize using environment variables (for PyTorch >= 1.5)
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        distributed_config['rank'] = rank
        distributed_config['world_size'] = world_size
        distributed_config['local_rank'] = local_rank
        distributed_config['distributed'] = True

        # Set device for each process
        torch.cuda.set_device(local_rank)
        dist_backend = distributed_config.get('backend', 'nccl')

        print(f"[Rank {rank}] Initializing distributed backend '{dist_backend}' with world size {world_size}")

        # Initialize the process group
        dist.init_process_group(
            backend=dist_backend,
            init_method='env://'
        )
        dist.barrier()
    else:
        # Single-process (non-distributed) training
        print('Distributed training not initialized. Running in single-process mode.')
        distributed_config['distributed'] = False
        distributed_config['rank'] = 0
        distributed_config['world_size'] = 1
        distributed_config['local_rank'] = 0

def cleanup():
    """
    Clean up the distributed training environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
