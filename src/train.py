import argparse
import time
import random
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from model.layers import *
from model.pangu_model import PanguModel
from era5_data.config import cfg
from era5_data.dataset import Era5Dataset
from era5_data.utils_data import load_variable_weights
from util import debug

def get_model_params_num(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script entry")
    parser.add_argument("--epoches", type=int, default=1, help="Number of epoches")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--distributed", action='store_true', help="Distributed training")

    args = parser.parse_args()
    return args


def train(args, model, device):
    data_path = cfg.PG_INPUT_PATH
    dataset = Era5Dataset(data_path,
                          begin_date='20070101',
                          end_date='20070102',
                          freq='1h',
                          lead_time=1,
                          normalize_data_path=os.path.join(data_path, 'ext_data', 'data_statistics.npz'))
    train_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(dataset)
    train_loader = DataLoader(dataset, batch_size=args.batch_size,
                              shuffle=False, sampler=train_sampler)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.PG.TRAIN.LR,
                                 weight_decay=cfg.PG.TRAIN.WEIGHT_DECAY)
    criterion = nn.L1Loss(reduction='none')

    upper_weights, surface_weights = load_variable_weights(device)

    model.train()
    for epoch in range(args.epoches):
        epoch_loss = 0
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        for batch, (input_upper, input_surface, target_upper, target_surface) in enumerate(train_loader):
            print(f"Epoch {epoch}, Batch {batch} running...")
            input_upper = input_upper.to(device)
            input_surface = input_surface.to(device)
            target_upper = target_upper.to(device)
            target_surface = target_surface.to(device)

            optimizer.zero_grad()
            output_upper, output_surface = model(input_upper, input_surface)
            loss_upper = criterion(output_upper, target_upper)
            weighted_loss_upper = torch.mean(loss_upper * upper_weights)

            loss_surface = criterion(output_surface, target_surface)
            weighted_loss_surface = torch.mean(loss_surface * surface_weights)
            loss = weighted_loss_upper + weighted_loss_surface * 0.25

            loss.backward()
            if args.distributed:
                rank = dist.get_rank()
                print(f"    Rank: {rank}, Loss: {loss.item():.4f}")
            else:
                print(f"    Loss: {loss.item():.4f}")
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")


def ddp_setup(args: argparse.Namespace):
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

    # TODO
    backend = 'gloo' # nccl, hccl

    debug(f"Init process group, backend: {backend}")
    if 'WORLD_SIZE' not in os.environ or 'RANK' not in os.environ:
        raise RuntimeError("WORLD_SIZE and RANK must be set.")

    dist.init_process_group(
        backend=backend
    )
    debug("Init process group done.")

# 单机多卡
# torchrun --nproc_per_node=2 train.py --distributed --batch_size=1

# 多机多卡
# master: (node-rank 0 --master-addr, --master-port)
# torchrun --nproc-per-node=4 --nnodes=2 --node-rank=0 train.py --distributed  \
#        --master-addr=?? --master-port=29500
def main():
    args = parse_args()
    dev = 'cpu'
    if torch.cuda.is_available():
        dev = 'cuda'
    dev = 'cpu'
    device = torch.device(dev)
    debug('device:', device)

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.distributed:
        debug("DDP setup")
        ddp_setup(args)

    model = PanguModel(device=device).to(device)
    if args.distributed:
        model = DDP(model)

    params_num = get_model_params_num(model)
    debug(f'params_num: {params_num:,}')

    train(args, model, device)

    if args.distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
