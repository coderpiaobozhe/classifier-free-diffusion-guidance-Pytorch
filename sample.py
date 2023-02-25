import os
import torch
import argparse
from unet import Unet
from dataloader_cifar import transback
from diffusion import GaussianDiffusion
from utils import get_named_beta_schedule
from embedding import ConditionalEmbedding
from torchvision.utils import save_image, make_grid
@torch.no_grad()
def sample(params):
    # load models
    net = Unet(in_ch = params.inch,
                mod_ch = params.modch,
                out_ch = params.outch,
                ch_mul = params.chmul,
                num_res_blocks = params.numres,
                cdim = params.cdim,
                use_conv=params.useconv,
                droprate = params.droprate,
                # num_heads = params.numheads,
                dtype=params.dtype).to(params.device)
    net.load_state_dict(torch.load(os.path.join(params.moddir, f'ckpt_{params.epc}_diffusion.pt')))
    cemblayer = ConditionalEmbedding(10, params.cdim, params.cdim).to(params.device)
    cemblayer.load_state_dict(torch.load(os.path.join(params.moddir, f'ckpt_{params.epc}_cemblayer.pt')))
    # settings for diffusion model
    betas = get_named_beta_schedule(num_diffusion_timesteps = params.T)
    diffusion = GaussianDiffusion(dtype = params.dtype,
                               model = net,
                               betas = betas,
                               w = params.w,
                               v = params.v,
                               device = params.device)
    # eval mode
    diffusion.model.eval()
    cemblayer.eval()
    # label settings
    if params.label == 'range':
        # assert params.end - params.start == 10
        lab = torch.ones(params.batchsize // 8, 8).type(torch.long) \
            * torch.arange(start = 0, end = 10).reshape(-1,1)
        lab = lab.reshape(-1,1).squeeze()
        lab = lab.to(params.device)
    else:
        lab = torch.randint(low = 0, high = 10, size = (params.batchsize,), device=params.device)
    # get label embeddings
    print(lab)
    cemb = cemblayer(lab)
    genshape = (params.batchsize, 3, 32, 32)
    generated = diffusion.sample(genshape, cemb = cemb)
    # transform samples into images
    img = transback(generated)
    # save images
    save_image(img, os.path.join(params.samdir, f'sample_{params.epc}_pict_{params.w}.png'), nrow = 8)
def main():
    # several hyperparameters for models
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize',type=int,default=80,help='batch size for training Unet model')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--w',type=float,default=1.8,help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=1.0,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epc',type=int,default=100,help='epochs for loading models')
    parser.add_argument('--cdim',type=int,default=10,help='dimension of conditional embedding')
    # parser.add_argument('--start',type=int,default=0)
    # parser.add_argument('--end',type=int,default=10)
    parser.add_argument('--device',default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),help='devices for training Unet model')
    parser.add_argument('--label',type=str,default='range',help='labels of generated images')
    parser.add_argument('--moddir',type=str,default='model',help='model addresses')
    parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')
    parser.add_argument('--inch',type=int,default=3,help='input channels for Unet model')
    parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
    parser.add_argument('--outch',type=int,default=3,help='output channels for Unet model')
    parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
    parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
    parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
    parser.add_argument('--droprate',type=float,default=0,help='dropout rate for model')
    # parser.add_argument('--numheads',type=int,default=1,help='number of attention heads')

    args = parser.parse_args()
    sample(args)

if __name__ == '__main__':
    main()
