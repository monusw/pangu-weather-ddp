from torch import nn
import torch
from collections import OrderedDict
import torch.utils.checkpoint as checkpoint

from model.layers import *
from model.pangu_model import PanguModel
from util import debug

def get_model_params_num(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    dev = 'cpu'
    if torch.cuda.is_available():
      dev = 'cuda'
    dev = 'cpu'
    debug('device:', dev)
    device = torch.device(dev)

    model = PanguModel(device=device).to(device)

    params_num = get_model_params_num(model)
    debug(f'params_num: {params_num:,}')

    x_surface = torch.randn((1, 4, 721, 1440)).to(device)
    x_upper = torch.randn((1, 5, 13, 721, 1440)).to(device)

    debug('x_upper:', x_upper.shape)
    debug('x_surface:', x_surface.shape)

    output, output_surface = model(x_upper, x_surface)
    # print(output)
    debug('out_upper', output.shape)
    debug('out_surface', output_surface.shape)

if __name__ == '__main__':
    main()
