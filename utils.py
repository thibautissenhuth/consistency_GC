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


# Scheduling functions for consistency models are taken from
# https://github.com/Kinyugo/consistency_models, originally licensed under the MIT License with the following notice.

# MIT License

# Copyright (c) 2023 Kinyugo Maina

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import torch
import torchvision
from ortools.graph.python import linear_sum_assignment
import numpy as np
from scipy import linalg
import math
from typing import List

class Train_config:
    def __init__(self, dataset: str='cifar10', n_train_steps: int=10000,
            loss_type: str='huber',
            diffusion_type: str='interpolation', sigma_min: float=0.001, sigma_max: float=50.,
            sigma_data: float=0.5, s0: int=10, s1: int=1280, ema_decay: float=0.999,
            ema_start: int=10000, minibatch_OT: bool=False, path: str | None=None,  generator_induced_traj: bool=False,
            generator_induced_traj_ema: bool=False, start_generator_induced_traj: int=0,
            end_generator_induced_traj: int=10000000, mix_gen_induced_traj: float=0.,
            mix_gen_induced_traj_end: float | None=None):

        self.dataset = dataset
        self.n_train_steps = n_train_steps
        self.loss_type = loss_type
        self.diffusion_type = diffusion_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.s0 = s0
        self.s1 = s1
        self.minibatch_OT = minibatch_OT
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.path = path
        self.generator_induced_traj = generator_induced_traj
        self.generator_induced_traj_ema = generator_induced_traj_ema
        self.start_generator_induced_traj = start_generator_induced_traj
        self.end_generator_induced_traj = end_generator_induced_traj
        self.mix_gen_induced_traj = mix_gen_induced_traj
        if mix_gen_induced_traj_end is None:
            self.mix_gen_induced_traj_end = mix_gen_induced_traj
        else:
            self.mix_gen_induced_traj_end = mix_gen_induced_traj_end

def get_mix_value(current_training_step, total_training_steps, start_value, end_value):
    perc_training = float(current_training_step) / float(total_training_steps)
    return (end_value - start_value) * perc_training + start_value

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def loss_image(set_A, set_B, cfg):
    if cfg.loss_type == 'huber':
        b, c, h, w = set_A.shape
        c = 0.00054 * np.sqrt(c * h * w) #0.03
        set_A = set_A.view(len(set_A),-1)
        set_B = set_B.view(len(set_B),-1)
        dists = torch.sqrt(((set_A - set_B)**2).sum(dim=-1) + c**2) - c
    elif cfg.loss_type == 'l2':
        set_A = set_A.view(len(set_A),-1)
        set_B = set_B.view(len(set_B),-1)
        dists = ((set_A - set_B)**2).sum(dim=-1)
    elif cfg.loss_type == 'lpips':
        dists = loss_lpips(set_A, set_B)
    return dists

def get_sigmas_karras(num_timesteps, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    '''ramp = torch.linspace(0, 1, int(n))
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.flip(sigmas, dims=(0,))'''

    rho_inv = 1.0 / rho
    # Clamp steps to 1 so that we don't get nans
    steps = torch.arange(num_timesteps, device=device) / max(num_timesteps - 1, 1)
    sigmas = sigma_min**rho_inv + steps * (
        sigma_max**rho_inv - sigma_min**rho_inv
    )
    sigmas = sigmas**rho
    return sigmas

def improved_timesteps_schedule(current_training_step, total_training_steps, initial_timesteps = 10, final_timesteps = 1280):
    """Implements the improved timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    total_training_steps_prime = math.floor(
        total_training_steps
        / (math.log2(math.floor(final_timesteps / initial_timesteps)) + 1)
    )
    num_timesteps = initial_timesteps * math.pow(
        2, math.floor(current_training_step / total_training_steps_prime)
    )
    num_timesteps = min(num_timesteps, final_timesteps) + 1
    return num_timesteps

def lognormal_timestep_distribution(num_samples, sigmas, mean = -1.1, std = 2.0):
    """Draws timesteps from a lognormal distribution.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw.
    sigmas : Tensor
        Standard deviations of the noise.
    mean : float, default=-1.1
        Mean of the lognormal distribution.
    std : float, default=2.0
        Standard deviation of the lognormal distribution.

    Returns
    -------
    Tensor
        Timesteps drawn from the lognormal distribution.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
        (torch.log(sigmas[:-1]) - mean) / (std * math.sqrt(2))
    )
    timesteps = torch.multinomial(pdf, num_samples, replacement=True)

    return timesteps

def improved_loss_weighting(sigmas):
    """Computes the weighting for the consistency loss.

    Parameters
    ----------
    sigmas : Tensor
        Standard deviations of the noise.

    Returns
    -------
    Tensor
        Weighting for the consistency loss.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    return 1 / (sigmas[1:] - sigmas[:-1])
'''
@torch.no_grad()
def eval_fid(data, generator, n_iter_FID, fid_metric, train_config):
    for i_FID_score in range(n_iter_FID):
        print(f"Num iter : {i_FID_score}/{n_iter_FID}")
        z = torch.randn_like(data)
        if train_config.diffusion_type == 'var_exp':
            steps = torch.ones((len(data))).to(data.device)
            sigmas_i = steps.float() * train_config.sigma_max
            z = z * sigmas_i.view(sigmas_i.shape[0],1,1,1)
            generations = generator(z, sigmas_i, augment_labels)
        else:
            steps = torch.ones((len(data))).to(data.device) + 0.0001
            generations = generator(z,steps)
        generations = (torch.clip(generations,-1,1) + 1) / 2
        fid_metric.update(generations,real=False)
    print('before sync')
    fid_metric.sync()
    print('after sync')
    #dist.barrier()
    if dist.get_rank() == 0:
        fid = fid_metric.compute()
        fid_metric.reset()
        return fid
    else:
        return None
'''

@torch.no_grad()
def eval_fid(data, generator, n_iter_FID, inception_net, real_data_features, train_config):
    fake_data_features = []
    for i in range(n_iter_FID):
        z = torch.randn_like(data)
        steps = torch.ones((len(data))).to(data.device)
        sigmas_i = steps.float() * train_config.sigma_max
        z = z * sigmas_i.view(sigmas_i.shape[0],1,1,1)
        generations = generator(z, sigmas_i)
        min_g, max_g = torch.min(generations).item(), torch.max(generations).item()
        generations = 2 * (generations - min_g) / (max_g - min_g) - 1 #torch.clip(generations,-1,1) 
        fake_data_features.append(get_inception_features(generations, inception_net)) 
    
    fake_data_features = torch.cat(fake_data_features, dim=0)[:real_data_features.shape[0]].cpu().numpy()
    
    mu_real, sigma_real = np.mean(real_data_features, axis=0), np.cov(real_data_features, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_data_features, axis=0), np.cov(fake_data_features, rowvar=False)
    
    fid_value = calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
    
    return fid_value

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"FID Computation.. Imaginary component {m}")
            #raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_inception_features(batch, model):
    with torch.no_grad():
        pred = model(batch)#[0]
    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    #if pred.size(2) != 1 or pred.size(3) != 1:
    #    pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
    #pred = pred.squeeze(3).squeeze(2)
    return pred

def return_index_matching(set_A, set_B):
    N_points = len(set_A)
    costs = torch.cdist(set_A.view(N_points,-1),set_B.view(N_points,-1),p=2)
    costs_torch = costs.cpu()
    costs = costs.cpu().numpy()
    end_nodes_unraveled, start_nodes_unraveled = np.meshgrid(
        np.arange(costs.shape[1]), np.arange(costs.shape[0])
    )
    start_nodes = start_nodes_unraveled.ravel()
    end_nodes = end_nodes_unraveled.ravel()
    arc_costs = costs.ravel()
    assignment = linear_sum_assignment.SimpleLinearSumAssignment()
    assignment.add_arcs_with_cost(start_nodes, end_nodes, arc_costs)
    status = assignment.solve()
    ind_j = np.array([assignment.right_mate(i) for i in range(assignment.num_nodes())])

    ind_j = torch.from_numpy(ind_j)
    ind_i = torch.arange(N_points)
    return ind_i, ind_j, costs_torch[ind_i, ind_j]

def get_next_batch(dataloader,dataloader_iterator):
    try:
        data = next(dataloader_iterator)
    except StopIteration:
        dataloader_iterator = iter(dataloader)
        data = next(dataloader_iterator)
    return data, dataloader_iterator
