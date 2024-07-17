from contextlib import contextmanager
import os
from torch.distributed import barrier

from torch.multiprocessing.spawn import spawn
from torch.distributed import init_process_group, get_rank, get_world_size
from time import sleep

@contextmanager
def main_process_first():
    if int(os.environ.get('WORLD_SIZE', '1')) == 1:
        yield
        return
    local_rank = int(os.environ['LOCAL_RANK'])
    try:
        if local_rank != 0:
            print(f"{local_rank} waiting for main")
            barrier()
        yield
    finally:
        print(f"{local_rank} common")
        barrier()
        if local_rank == 0:
            print(f"{local_rank} waiting for others")
            barrier()

def get_local_rank():
    return int(os.environ['LOCAL_RANK'])

def worker_main(
    local_rank,
    nodeid,
    nnodes,
    nprocs,
    master_addr,
    master_port,
    fn,
    args
):
    world_size = nnodes * nprocs
    rank = nodeid * nnodes + local_rank
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['MASTER_ADDR'] = str(master_addr)

    print(f"Hello from {get_local_rank()}")
    
    init_process_group()
    fn(*args)

@main_process_first()
def collective_function():
    if 0 == get_local_rank():
        print(f"Baking {get_local_rank()}")
        sleep(5)
    else:
        print(f"Already baking {get_local_rank()}")

def main():
    nodeid = 0
    nnodes = 1
    nprocs = 4
    master_addr = 'localhost'
    master_port = 29501
    args = (
        nodeid,
        nnodes,
        nprocs,
        master_addr,
        master_port,
        collective_function,
        tuple()
    )

    spawn(worker_main, args, nprocs=nprocs, join=True)
    print("All childern exited")
    
if __name__ == '__main__':
    main()
    