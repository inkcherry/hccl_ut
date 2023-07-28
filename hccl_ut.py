import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
import habana_frameworks.torch.distributed.hccl

import habana_frameworks.torch.core as htcore
import torch.distributed as dist


# torch.distributed.broadcast(flags,
#                             mpu.get_tensor_model_parallel_src_rank(),
#                             group=mpu.get_tensor_model_parallel_group(),
#                             async_op=args.use_hpu)


device = torch.device('hpu')
def rank_print(str):
    print(f"rank:{dist.get_rank()}: {str}")
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["ID"] = str(rank)
    #distributed package for HCCL
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    # dist.init_process_group(backend='hccl')


def cleanup():
    dist.destroy_process_group()

# int tresh_hold=100

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    print(f"finish set up")
    rank = int(dist.get_rank())
    mp_ranks = range(rank,rank+1)
    # print(mp_ranks)
    # all_group = torch.distributed.new_group(mp_ranks)
    # dist.barrier()

    # goups_list=[]
    # for i in range(tresh_hold):
        
    all_group = torch.distributed.new_group(mp_ranks)
    # dist.barrier()

    all_group2 = torch.distributed.new_group(mp_ranks)
    # dist.barrier()

    all_group3 = torch.distributed.new_group(mp_ranks)

    # dist.barrier()

    all_group4 = torch.distributed.new_group(mp_ranks)
    # dist.barrier()

    all_group5 = torch.distributed.new_group(mp_ranks)
    # dist.barrier()

    all_group6 = torch.distributed.new_group(mp_ranks)

    # dist.barrier()
    all_group7= torch.distributed.new_group(mp_ranks)
    # dist.barrier()


    rank_print(f"new_group setup")
    x = torch.rand(50304, 3072) + rank

    x = x.to("hpu:{}".format(rank))
    rank_print(f"before broadcast")
    rank_print("b1"+str(all_group))

    dist.broadcast(x,rank,group=all_group)
    rank_print("b2")

    dist.broadcast(x,rank,group=all_group2)
    rank_print("b3")

    dist.broadcast(x,rank,group=all_group3)
    rank_print("b4")

    dist.broadcast(x,rank,group=all_group4)
    rank_print("b5")


    dist.broadcast(x,rank,group=all_group5)
    rank_print("b6")


    dist.broadcast(x,rank,group=all_group6)
    rank_print("b7")

    # this line will be ok `dist.broadcast(x,rank,group=all_group)`
    # the fllowing will get comm init error,  break max threshhold
    # dist.destroy_process_group()

    dist.broadcast(x,rank,group=all_group7)

    rank_print(f"after broadcast")
    rank_print("just test")
    

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    world_size = 5
    run_demo(demo_basic, world_size)
