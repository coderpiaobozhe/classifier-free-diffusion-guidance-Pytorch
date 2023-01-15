import os
import torch
import argparse
import numpy as np
from torch import nn
from unet import Unet
from tqdm import tqdm
import torch.optim as optim
from diffusion import GaussianDiffusion
from utils import get_named_beta_schedule
from embedding import ConditionalEmbedding
from Scheduler import GradualWarmupScheduler
from dataloader_cifar import load_data, transback
from torchvision.utils import save_image, make_grid

def train(params):
    #initialize settings
    dataloader = load_data(params)
    net = Unet(in_ch = params.inch,
                mod_ch = params.modch,
                out_ch = params.outch,
                ch_mul = params.chmul,
                num_res_blocks = params.numres,
                cdim = params.cdim,
                use_conv=params.useconv,
                droprate = params.droprate,
                num_heads = params.numheads,
                dtype=params.dtype)
    betas = get_named_beta_schedule(num_diffusion_timesteps = params.T)
    optimizer = torch.optim.AdamW(net.parameters(),
                                lr = params.lr,
                                weight_decay = 1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                T_max = params.epoch,
                                eta_min = 0,
                                last_epoch = -1)
    warmUpScheduler = GradualWarmupScheduler(optimizer = optimizer,
                                multiplier = params.multiplier,
                                warm_epoch = params.epoch // 10,
                                after_scheduler = cosineScheduler)
    diffusion = GaussianDiffusion(dtype = params.dtype,
                                model = net,
                                betas = betas,
                                w = params.w,
                                v = params.v,
                                device = params.device)
    cemblayer = ConditionalEmbedding(10, params.cdim, params.cdim).to(params.device)
    # training
    for epc in range(params.epoch):
        diffusion.model.train()
        cemblayer.train()
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for img, lab in tqdmDataLoader:
                b = img.shape[0]
                optimizer.zero_grad()
                x_0 = img.to(params.device)
                lab = lab.to(params.device)
                cemb = cemblayer(lab)
                cemb[np.where(np.random.rand(b)<params.threshold)] = 0
                loss = diffusion.trainloss(x_0,{'cemb':cemb}) / (b**2)
                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": epc,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        # evaluation
        diffusion.model.eval()
        cemblayer.eval()
        # generating samples
        with torch.no_grad():
            lab = torch.randint(low = 0, high = 10, size = (params.batchsize,), device=params.device)
            cemb = cemblayer(lab)
            genshape = (params.batchsize, 3, 32, 32)
            generated = diffusion.sample(genshape,{'cemb':cemb})
            img = transback(generated)
        image = make_grid(img, params.batchsize // 8, 0)
        save_image(image, os.path.join(params.samdir, f'generated_{epc}_pict.png'))
        torch.save(diffusion.model.state_dict(), os.path.join(params.moddir, f'ckpt_{epc}_diffusion.pt'))
        torch.save(cemblayer.state_dict(), os.path.join(params.moddir, f'ckpt_{epc}_cemblayer.pt'))
        torch.cuda.empty_cache()
  
def main():
    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize',type=int,default=128,help='batch size for training Unet model')
    parser.add_argument('--numworkers',type=int,default=4,help='num workers for training Unet model')
    parser.add_argument('--inch',type=int,default=3,help='input channels for Unet model')
    parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    parser.add_argument('--outch',type=int,default=3,help='output channels for Unet model')
    parser.add_argument('--chmul',type=list,default=[1,2,4,8],help='architecture parameters training Unet model')
    parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
    parser.add_argument('--cdim',type=int,default=10,help='dimension of conditional embedding')
    parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
    parser.add_argument('--droprate',type=float,default=0,help='dropout rate for model')
    parser.add_argument('--numheads',type=int,default=1,help='number of attention heads')
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--lr',type=float,default=1e-4,help='learning rate')
    parser.add_argument('--w',type=float,default=1.8,help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=0.3,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch',type=int,default=20,help='epochs for training')
    parser.add_argument('--device',default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),help='devices for training Unet model')
    parser.add_argument('--multiplier',type=float,default=2.5,help='multiplier for warmup')
    parser.add_argument('--threshold',type=float,default=0.1,help='threshold for classifier-free guidance')
    parser.add_argument('--moddir',type=str,default='model',help='model addresses')
    parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()