# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import torch
from . import training_stats
import socket 

#----------------------------------------------------------------------------

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
        except OSError:
            return True
        else:
            return False


def init():
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        for port in range(29500, 29500 + 1000):
            if not is_port_in_use(port):
                os.environ['MASTER_PORT'] = str(port)
                break
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

    sync_device = torch.device('cuda') if get_world_size() > 1 else None
    training_stats.init_multiprocessing(rank=get_rank(), sync_device=sync_device)

#----------------------------------------------------------------------------

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

#----------------------------------------------------------------------------

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

#----------------------------------------------------------------------------

def should_stop():
    return False

#----------------------------------------------------------------------------

def update_progress(cur, total):
    _ = cur, total

#----------------------------------------------------------------------------

def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)

#----------------------------------------------------------------------------
