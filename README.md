<!-- # Copyright 2024 Thibaut Issenhuth, Ludovic Dos Santos, Jean-Yves Franceschi, Alain Rakotomamonjy

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->


## Improving Consistency Models with Generator-Induced Coupling

Official implementation of the paper *Improving Consistency Models with Generator-Induced Flows* (Thibaut Issenhuth, Sangchul Lee, Ludovic Dos Santos, Jean-Yves Franceschi, Chansoo Kim, Alain Rakotomamonjy).


## [Preprint](https://arxiv.org/abs/2406.09570) 


## Requirements

Python libraries: See [requirements.txt](./requirements.txt).

## Getting started

To train a base consistency model on CIFAR-10, run:

```.bash
torchrun --nproc_per_node=1 train_consistency.py --cfg cifar10_Base --device 0 --eval_fid 1 --eval_freq 5000
```

To train a consistency model with 50% generator-induced trajectories on CIFAR-10, run: 
```.bash
torchrun --nproc_per_node=1 train_consistency.py --cfg cifar10_GenInduced_ema_mix50 --device 0 --eval_fid 1 --eval_freq 5000
```

The *mix_gen_induced_traj* parameter in config file handles the percentage of generator-induced trajectories per batch. 

## References
If you find the code useful for your research, please consider citing:
```bib
@article{issenhuth2024improving,
  title={Improving Consistency Models with Generator-Induced Flows},
  author={Issenhuth, Thibaut and Lee, Sangchul and Santos, Ludovic Dos and Franceschi, Jean-Yves and Kim, Chansoo and Rakotomamonjy, Alain},
  journal={arXiv preprint arXiv:2406.09570},
  year={2024}
}
```
