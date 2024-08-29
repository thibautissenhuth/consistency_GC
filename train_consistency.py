# Copyright 2024 Thibaut Issenhuth, Ludovic Dos Santos, Jean-Yves Franceschi, Alain Rakotomamonjy

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torchvision

import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from datasets import load_dataset

from utils import  get_next_batch, eval_fid, loss_image \
                get_sigmas_karras, lognormal_timestep_distribution, improved_timesteps_schedule, improved_loss_weighting, \
                get_mix_value

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import argparse

from torchmetrics.image.fid import FrechetInceptionDistance
from ema_pytorch import EMA

import lpips
from lion_pytorch import Lion

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument("--cfg")
parser.add_argument("--eval_freq",type=int,default=1000)
parser.add_argument("--eval_fid",type=int,default=1)
parser.add_argument("--device",type=int,default=0)
parser.add_argument("--path",type=str,default=None)
args = parser.parse_args()

if args.eval_fid:
    print('Fid eval')
else:
    print('No Fid eval')

initialize(version_base=None,config_path="configs", job_name="test_app")
config = compose(config_name=args.cfg)

train_cfg = instantiate(config.train_config)
print(train_cfg)

save_path = os.path.join('exps',config.name_exp)
if not os.path.exists(save_path):
    os.makedirs(save_path)

print(OmegaConf.to_yaml(config))
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

if config.dataset == 'celeba':
    dataroot = "~/data/"
    transform = transforms.Compose([transforms.Resize(config.image_size, interpolation=torchvision.transforms.functional.InterpolationMode.LANCZOS),
                                    transforms.CenterCrop(config.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])
    train_data = torchvision.datasets.CelebA(dataroot, split='train', target_type='attr',
                                                   transform=transform, download=False)
    test_data = torchvision.datasets.CelebA(dataroot, split='test', target_type='attr',
                                                   transform=transform, download=False)
elif config.dataset == 'imagenet':
    def transform_hf(examples):
        transform = transforms.Compose([
                    transforms.Resize(config.image_size),
                    transforms.CenterCrop(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        examples["image"] = [transform(image.convert("RGB")) for image in examples["image"]]
        return examples
    dataset = load_dataset("imagenet-1k", num_proc=8)
    train_data, test_data = dataset['train'], dataset['test']
    train_data.set_format("torch")
    test_data.set_format("torch")
    train_data.set_transform(transform_hf)
    test_data.set_transform(transform_hf)
elif config.dataset == 'cifar10':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size,
                                            shuffle=True, num_workers=config.workers)
train_dataloader_iterator = iter(train_dataloader)

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size,
                                         shuffle=True, num_workers=config.workers)


train_dataloader_FID = train_dataloader
train_dataloader_iterator_FID = iter(train_dataloader_FID)
test_dataloader_FID = test_dataloader

if args.device == 0:
    device = torch.device("cuda:0")
elif args.device == 1:
    device = torch.device("cuda:1")

FID_metric = FrechetInceptionDistance(reset_real_features=False,normalize=True).to(device) ## Normalize=True -> img in [0,1]; False -> img in [0,255]
best_fid = 10000
n_iter_FID = 10000//config.batch_size
for i, data in enumerate(test_dataloader_FID, 0):
    if config.dataset == 'imagenet':
        real_data = data["image"].to(device)
    else:
        real_data = data[0].to(device)
    real_data = (real_data + 1)/2
    FID_metric.update(real_data,real=True)
    if i == n_iter_FID:
        break
if i<n_iter_FID:
    n_iter_FID = i

for i in range(n_iter_FID):
    data, train_dataloader_iterator_FID = get_next_batch(train_dataloader_FID, train_dataloader_iterator_FID)
    if config.dataset == 'imagenet':
        real_data = data["image"].to(device)
    else:
        real_data = data[0].to(device)
    real_data = (real_data + 1)/2
    FID_metric.update(real_data,real=False)
print("train/test FID = ",FID_metric.compute())
FID_metric.reset()

FID_train_metric = FrechetInceptionDistance(reset_real_features=False,normalize=True).to(device) ## Normalize=True -> img in [0,1]; False -> img in [0,255]
n_iter_FID_train = 50000//config.batch_size
for i, data in enumerate(train_dataloader_FID, 0):
    if config.dataset == 'imagenet':
        real_data = data["image"].to(device)
    else:
        real_data = data[0].to(device)
    real_data = (real_data + 1)/2
    FID_train_metric.update(real_data,real=True)
    if i == n_iter_FID_train:
        break
if i<n_iter_FID_train:
    n_iter_FID_train = i

unet = instantiate(config.generator)
unet = unet.to(device)

unet_ema = EMA(
    unet,
    beta = train_cfg.ema_decay,              # exponential moving average factor
    update_after_step = train_cfg.ema_start,    # only after this number of .update() calls will it start updating
    update_every = 5,          # how often to actually update, to save on compute (updates every 10th .update() call)
)

if train_cfg.path is not None:
    checkpoint = torch.load(train_cfg.path, map_location=torch.device(args.device))
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet_ema.load_state_dict(checkpoint['model_ema_state_dict'])

# Training Loop
if config.optimizer == 'adam':
    optimizer = optim.Adam(unet.parameters(), lr=config.lr)
elif config.optimizer == 'radam':
    optimizer = optim.RAdam(unet.parameters(), lr=config.lr)
elif config.optimizer == 'lion':
    optimizer = Lion(unet.parameters(), lr=config.lr)

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
loss_lpips = lpips.LPIPS(net='alex').to(device)

print("Starting Training...")
### TRAINING ####
for i_train in range(train_cfg.n_train_steps):
    unet.train()
    data, train_dataloader_iterator = get_next_batch(train_dataloader, train_dataloader_iterator)
    if config.dataset == 'imagenet':
        batch_real_data = data["image"].to(device)
    else:
        batch_real_data = data[0].to(device)
    b_size = batch_real_data.size(0)

    batch_z = torch.randn_like(batch_real_data)
    current_n_step = improved_timesteps_schedule(i_train, train_cfg.n_train_steps,
                                    initial_timesteps = train_cfg.s0, final_timesteps = train_cfg.s1)
    if train_cfg.diffusion_type == 'interpolation':
        sigmas = get_sigmas_karras(current_n_step, train_cfg.sigma_min, 80)
        steps = lognormal_timestep_distribution(len(batch_real_data), sigmas)
        loss_weights = improved_loss_weighting(sigmas)[steps].to(batch_real_data.device)
        sigmas_prop = sigmas / (sigmas + 1)
        steps_proportion = sigmas_prop[steps].view(len(batch_real_data),1,1,1).to(batch_real_data.device)
        steps_1_proportion = sigmas_prop[steps + 1].view(len(batch_real_data),1,1,1).to(batch_real_data.device)
        batch_z_i = batch_z * steps_proportion + batch_real_data * (1 - steps_proportion)
        if train_cfg.generator_induced_traj==True:
            mixing_value = get_mix_value(i_train, train_cfg.n_train_steps, \
                        train_cfg.mix_gen_induced_traj, train_cfg.mix_gen_induced_traj_end)
            mask = (torch.rand((b_size,1,1,1)) > mixing_value).to(device)
            with torch.no_grad():
                batch_real_data_standard = batch_real_data
                if train_cfg.generator_induced_traj_ema:
                    batch_real_data = unet_ema(batch_z_i, steps_proportion.flatten())
                else:
                    batch_real_data = unet(batch_z_i, steps_proportion.flatten())

                if mixing_value > 0.:
                    batch_real_data = mask * batch_real_data + ~mask * batch_real_data_standard

                batch_z_i = batch_z * steps_proportion + batch_real_data * (1 - steps_proportion)
        batch_z_ip1 = batch_z * steps_1_proportion + batch_real_data * (1 - steps_1_proportion)
    elif train_cfg.diffusion_type == 'var_exp':
        sigmas = get_sigmas_karras(current_n_step, train_cfg.sigma_min, train_cfg.sigma_max)
        steps = lognormal_timestep_distribution(len(batch_real_data), sigmas)
        loss_weights = improved_loss_weighting(sigmas)[steps].to(batch_real_data.device)
        sigmas_i = sigmas[steps].to(batch_real_data.device)
        sigmas_ip1 = sigmas[steps + 1].to(batch_real_data.device)
        batch_z_i = batch_real_data + sigmas_i.view(sigmas_i.shape[0],1,1,1) * batch_z

        if train_cfg.generator_induced_traj==True:
            mixing_value = get_mix_value(i_train, train_cfg.n_train_steps, \
                        train_cfg.mix_gen_induced_traj, train_cfg.mix_gen_induced_traj_end)
            mask = (torch.rand((b_size,1,1,1)) > mixing_value).to(device)
            with torch.no_grad():
                batch_real_data_standard = batch_real_data
                if train_cfg.generator_induced_traj_ema:
                    batch_real_data = unet_ema(batch_z_i, sigmas_i)
                else:
                    batch_real_data = unet(batch_z_i, sigmas_i)
                if mixing_value > 0.:
                    batch_real_data = mask * batch_real_data + ~mask * batch_real_data_standard
                batch_z_i = batch_real_data + sigmas_i.view(sigmas_i.shape[0],1,1,1) * batch_z

        batch_z_ip1 = batch_real_data + sigmas_ip1.view(sigmas_ip1.shape[0],1,1,1) * batch_z

    optimizer.zero_grad()
    rng_state = torch.cuda.get_rng_state(device)
    if train_cfg.diffusion_type == 'interpolation':
        with torch.no_grad():
            generations = unet(batch_z_i, steps_proportion.flatten())
        torch.cuda.set_rng_state(rng_state, device=device)
        generations_1 = unet(batch_z_ip1, steps_1_proportion.flatten())
        loss_batch = loss_image(generations, generations_1, train_cfg)
    elif train_cfg.diffusion_type == 'var_exp':
        with torch.no_grad():
            generations = unet(batch_z_i, sigmas_i)
        torch.cuda.set_rng_state(rng_state, device=device)
        generations_1 = unet(batch_z_ip1, sigmas_ip1)
        loss_batch = loss_image(generations, generations_1, train_cfg)

    loss = (loss_weights * loss_batch).mean()

    loss.backward()
    optimizer.step()
    unet_ema.update()

    if i_train % 50 == 0:
        print('[%d/%d]\tLoss: %.4f\t'
            % (i_train, train_cfg.n_train_steps,
                loss.item()),flush=True)

    if (i_train % args.eval_freq == 0) or (i_train == train_cfg.n_train_steps):
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        if train_cfg.diffusion_type == 'var_exp':
            ax.scatter(sigmas_i.flatten().detach().cpu().numpy(), loss_batch.flatten().detach().cpu().numpy(), alpha=0.5)
        elif train_cfg.diffusion_type == 'interpolation':
            ax.scatter(steps_proportion.flatten().detach().cpu().numpy(), loss_batch.flatten().detach().cpu().numpy(), alpha=0.5)
        fig.savefig(os.path.join(save_path,str(i_train)+'_loss_per_timestep.jpeg'))
        plt.close(fig)


        if args.eval_fid == 1:
            unet.eval()
            unet_ema.eval()
            print("fid : ", eval_fid(batch_real_data, unet, n_iter_FID, FID_metric, train_cfg), flush=True)
            fid_test_ema = eval_fid(batch_real_data, unet_ema, n_iter_FID, FID_metric, train_cfg)
            print("fid EMA: ", fid_test_ema, flush=True)
            print("fid train EMA: ", eval_fid(batch_real_data, unet_ema, n_iter_FID_train, FID_train_metric, train_cfg), flush=True)

            if fid_test_ema < best_fid:
                save_path_ckpt = os.path.join(save_path,'best_model.pt')
                best_fid = fid_test_ema
                torch.save({
                    'model_state_dict': unet.state_dict(),
                    'model_ema_state_dict': unet_ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_step': i_train,
                    }, save_path_ckpt)

        generations = torch.clip(generations,-1,1)

        torchvision.utils.save_image(generations,os.path.join(save_path,'train_'+str(i_train)+'.jpeg'),normalize=True)
        generations_1 = torch.clip(generations_1,-1,1)
        torchvision.utils.save_image(generations_1,os.path.join(save_path,'pred_'+str(i_train)+'.jpeg'),normalize=True)

        batch_z_ip1 = torch.clip(batch_z_ip1,-1,1)
        torchvision.utils.save_image(batch_z_ip1,os.path.join(save_path,'pred_inputs_'+str(i_train)+'.jpeg'),normalize=True)

        with torch.no_grad():
            z = torch.randn_like(batch_real_data)

            if train_cfg.diffusion_type == 'var_exp':
                steps = torch.zeros((len(batch_real_data))) + current_n_step - 1
                sigmas_i = sigmas[steps.long()].to(batch_real_data.device)
                sigmas_i = sigmas_i.view(sigmas_i.shape[0], 1, 1, 1)
                generations = unet(z * sigmas_i, sigmas_i)
                generations_ema = unet_ema(z * sigmas_i, sigmas_i)
            elif train_cfg.diffusion_type == 'interpolation':
                steps = torch.ones((len(batch_real_data))).to(batch_real_data.device)
                generations = unet(z,steps)
                generations_ema = unet_ema(z,steps)

            generations = torch.clip(generations,-1,1)
            generations_ema = torch.clip(generations_ema,-1,1)
        torchvision.utils.save_image(generations,os.path.join(save_path,'generations_'+str(i_train)+'.jpeg'),normalize=True)
        torchvision.utils.save_image(generations_ema,os.path.join(save_path,'generations_ema_'+str(i_train)+'.jpeg'),normalize=True)
