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
    net = Unet().to(params.device)
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
        assert params.end - params.start == 8
        lab = torch.ones(8, params.batchsize // 8).type(torch.int) \
            * torch.arange(start = params.start, end = params.end).reshape(-1, 1)
        lab = lab.reshape(-1,1).squeeze()
        lab = lab.to(params.device)
    else:
        lab = torch.randint(low = 0, high = 10, size = (params.batchsize,), device=params.device)
    # get label embeddings
    cemb = cemblayer(lab)
    genshape = (params.batchsize, 3, 32, 32)
    generated = diffusion.sample(genshape,{'cemb':cemb})
    # transform samples into images
    img = transback(generated)
    # save images
    image = make_grid(img, params.batchsize // 8, 0)
    save_image(image, os.path.join(params.samdir, f'sample_{params.epc}_pict.png'))
def main():
    # several hyperparameters for models
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize',type=int,default=128,help='batch size for training Unet model')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--w',type=float,default=1.8,help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=0.3,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epc',type=int,default=0,help='epochs for loading models')
    parser.add_argument('--cdim',type=int,default=10,help='dimension of conditional embedding')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end',type=int,default=8)
    parser.add_argument('--device',default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),help='devices for training Unet model')
    parser.add_argument('--label',type=str,default='range',help='labels of generated images')
    parser.add_argument('--moddir',type=str,default='model',help='model addresses')
    parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')

    args = parser.parse_args()
    sample(args)

if __name__ == '__main__':
    main()