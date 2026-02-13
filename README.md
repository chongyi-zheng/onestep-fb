<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Can We Really Learn One Representation to Optimize All Rewards?</h1>
      <h2>
        <a href="https://arxiv.org/abs/2602.11399">Paper</a> &emsp; 
        <a href="https://chongyi-zheng.github.io/onestep-fb">Website</a>
        <a href="">Thread</a>
      </h2>
    </summary>
  </ul>
</div>

<img src="assets/teaser.svg" width="95%">

</div>

# Overview

One-step Forward-Backward Representation Learning (one-step FB) is an *unsupervised pre-training* methods for RL that enables efficient zero-shot adaptation and fine-tuning on downstream tasks. Our method is simpler than [Forward-Backward Representation Learning (FB)](https://arxiv.org/abs/2103.07945).

This repository contains code for running the one-step FB algorithm and 5 baselines in the offline setting. These baselines include [Laplacian](https://arxiv.org/abs/1810.04586), [BYOL-&gamma;](https://arxiv.org/abs/2506.10137), [ICVF](https://arxiv.org/abs/2304.04782), [HILP](https://arxiv.org/abs/2402.15567), [FB](https://arxiv.org/abs/2103.07945).

# Installation

1. Create an Anaconda environment: `conda create -n onestep-fb python=3.10.13 -y`
2. Activate the environment: `conda activate onestep-fb`
3. Install the dependencies:
   ```
   conda install -c conda-forge glew -y
   conda install -c conda-forge mesalib -y
   pip install -r requirements.txt
   ```
4. Export environment variables
   ```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia:$HOME/.mujoco/mujoco210/bin
   export PYTHONPATH=path_to_onestep_fb_dir
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   ```

# Datasets

## ExORL

The default directory to store the ExORL datasets is `~/.exorl`.

<details>
<summary><b>Click to expand the commands to generate ExORL datasets</b></summary>

<b></b>

1. Download exploratory datasets.

    ```shell
    sh ./data_gen_scripts/exorl_download.sh cheetah rnd
    sh ./data_gen_scripts/exorl_download.sh walker rnd
    sh ./data_gen_scripts/exorl_download.sh quadruped rnd
    sh ./data_gen_scripts/exorl_download.sh jaco rnd
    ```

2. Generate pre-training datasets.

    This may take 10 - 20 mins for each dataset. You can decrease the number of subprocesses by changing `num_workers`, but it will take longer to generate the datasets.

    ```shell
    # walker
    python data_gen_scripts/generate_exorl_dataset.py --domain_name=walker --save_path=~/.exorl/data/rnd-walker.hdf5 --num_workers=16

    # cheetah
    python data_gen_scripts/generate_exorl_dataset.py --domain_name=cheetah --save_path=~/.exorl/data/rnd-cheetah.hdf5 --num_workers=16

    # quadruped
    python data_gen_scripts/generate_exorl_dataset.py --domain_name=quadruped --save_path=~/.exorl/data/rnd-quadruped.hdf5 --num_workers=16

    # jaco
    python data_gen_scripts/generate_exorl_dataset.py --domain_name=jaco --save_path=~/.exorl/data/rnd-jaco.hdf5 --num_workers=16
    ```
  
3. Generate zero-shot adaptation datasets.
    
    This may take 2 - 5 mins for each dataset.

    ```shell
    # walker
    python data_gen_scripts/generate_exorl_dataset.py --domain_name=walker --save_path=~/.exorl/data/rnd-walker-val.hdf5 --num_workers=16 --skip_size=5_000_000 --dataset_size=100_000

    # cheetah
    python data_gen_scripts/generate_exorl_dataset.py --domain_name=cheetah --save_path=~/.exorl/data/rnd-cheetah-val.hdf5 --num_workers=16 --skip_size=5_000_000 --dataset_size=100_000

    # quadruped
    python data_gen_scripts/generate_exorl_dataset.py --domain_name=quadruped --save_path=~/.exorl/data/rnd-quadruped-val.hdf5 --num_workers=16 --skip_size=5_000_000 --dataset_size=100_000

    # jaco
    python data_gen_scripts/generate_exorl_dataset.py --domain_name=jaco --save_path=~/.exorl/data/rnd-jaco-val.hdf5 --num_workers=16 --skip_size=5_000_000 --dataset_size=100_000
    ```

</details>

## OGBench

We use the default datasets in OGBench for pre-training and zero-shot adaptation. The following datasets will be downloaded automatically to `~/.ogbench/data` when executing the code:

<details>
<summary><b>Click to expand the list of OGBench datasets</b></summary>

- antmaze large navigate: `antmaze-large-navigate-v0.npz` and `antmaze-large-navigate-v0-val.npz`.
- antmaze teleport navigate: `antmaze-teleport-navigate-v0.npz` and `antmaze-teleport-navigate-v0-val.npz`.
- cube single play: `cube-single-play-v0.npz` and `cube-single-play-v0-val.npz`.
- scene play: `scene-play-v0.npz` and `scene-play-v0-val.npz`.
- visual cube single play: `visual-cube-single-noisy-v0.npz` and `visual-cube-single-noisy-v0-val.npz`.
- visual scene play: `visual-scene-play-v0.npz` and `visual-scene-play-v0-val.npz`.

</details>

<b></b>

# Running experiments

Check the `agents` folder for available algorithms and default hyperparameters. 

> [!NOTE]
> For state-based tasks, using 8 CPU, 16GB CPU memory, and a single A6000 (or better) GPU is enough. 
> For image-based tasks, we need 20 CPU, 128GB CPU memory, and a single A6000 (or better) GPU.


Here are some example commands to run experiments:

For full list of commands on each domain or task, see below.

## One-step FB

<details>
<summary><b>Click to expand the full list of commands</b></summary>

### Offline pre-training

```shell
# exorl walker
python main.py --env_name=exorl-rnd-walker --agent=agents/onestep_fb.py --agent.backward_repr_hidden_dims="(256,256,256)" --agent.forward_repr_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.q_agg=mean --agent.orthonorm_coeff=0.1 --agent.alpha=0 --agent.tanh_squash=False

# exorl cheetah
python main.py --env_name=exorl-rnd-cheetah --agent=agents/onestep_fb.py --agent.backward_repr_hidden_dims="(256,256,256)" --agent.forward_repr_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.reward_temperature=3 --agent.q_agg=mean --agent.orthonorm_coeff=1 --agent.alpha=0 --agent.tanh_squash=False

# exorl quadruped
python main.py --env_name=exorl-rnd-quadruped --agent=agents/onestep_fb.py --agent.backward_repr_hidden_dims="(256,256,256)" --agent.forward_repr_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.orthonorm_coeff=0.03 --agent.alpha=0

# exorl jaco
python main.py --env_name=exorl-rnd-jaco --agent=agents/onestep_fb.py --agent.backward_repr_hidden_dims="(256,256,256)" --agent.forward_repr_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.orthonorm_coeff=0.03 --agent.alpha=0

# antmaze large navigate
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/onestep_fb.py --agent.alpha=0.03

# antmaze teleport navigate
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/onestep_fb.py --agent.alpha=0.1

# cube single play
python main.py --env_name=ogbench-cube-single-play-v0 --agent=agents/onestep_fb.py --agent.reward_temperature=300 --agent.alpha=0.3

# scene play
python main.py --env_name=ogbench-scene-play-v0 --agent=agents/onestep_fb.py --agent.reward_temperature=300 --agent.alpha=0.3

# visual cube single play
python main.py --env_name=visual-cube-single-play-v0 --agent=agents/onestep_fb.py --agent.batch_size=256 --agent.forward_repr_layer_norm=False --agent.backward_repr_layer_norm=False --agent.reward_temperature=300 --agent.encoder=impala_small --agent.dataset.p_aug=0.5 --agent.dataset.frame_stack=3

# visual scene play
python main.py --env_name=visual-scene-play-v0 --agent=agents/onestep_fb.py --agent.batch_size=256 --agent.forward_repr_layer_norm=False --agent.backward_repr_layer_norm=False --agent.encoder=impala_small --agent.dataset.p_aug=0.5 --agent.dataset.frame_stack=3
```

</details>

## Prior methods

### Laplacian

<details>
<summary><b>Click to expand the full list of commands</b></summary>

```shell
# exorl walker
python main.py --env_name=exorl-rnd-walker --agent=agents/laplacian.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.orthonorm_coeff=1 --agent.alpha=30

# exorl cheetah
python main.py --env_name=exorl-rnd-cheetah --agent=agents/laplacian.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.orthonorm_coeff=1 --agent.alpha=10

# exorl quadruped
python main.py --env_name=exorl-rnd-quadruped --agent=agents/laplacian.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.orthonorm_coeff=1 --agent.alpha=10

# exorl jaco
python main.py --env_name=exorl-rnd-jaco --agent=agents/laplacian.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.orthonorm_coeff=1 --agent.alpha=30

# antmaze large navigate
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/laplacian.py --agent.orthonorm_coeff=0.1

# antmaze teleport navigate
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/laplacian.py --agent.alpha=30

# cube single play
python main.py --env_name=ogbench-cube-single-play-v0 --agent=agents/laplacian.py

# scene play
python main.py --env_name=ogbench-scene-play-v0 --agent=agents/laplacian.py
```

</details>

### BYOL-&gamma;

<details>
<summary><b>Click to expand the full list of commands</b></summary>

```shell
# exorl walker
python main.py --env_name=exorl-rnd-walker --agent=agents/byol_gamma.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.orthonorm_coeff=0.01 --agent.alpha=10

# exorl cheetah
python main.py --env_name=exorl-rnd-cheetah --agent=agents/byol_gamma.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.orthonorm_coeff=0.01

# exorl quadruped
python main.py --env_name=exorl-rnd-quadruped --agent=agents/byol_gamma.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.orthonorm_coeff=0.01

# exorl jaco
python main.py --env_name=exorl-rnd-jaco --agent=agents/byol_gamma.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.orthonorm_coeff=1 --agent.alpha=1

# antmaze large navigate
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/byol_gamma.py

# antmaze teleport navigate
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/byol_gamma.py --agent.alpha=1

# cube single play
python main.py --env_name=ogbench-cube-single-play-v0 --agent=agents/byol_gamma.py --agent.alpha=30

# scene play
python main.py --env_name=ogbench-scene-play-v0 --agent=agents/byol_gamma.py

# visual cube single play
python main.py --env_name=visual-cube-single-play-v0 --agent=agents/byol_gamma.py --agent.batch_size=256 --agent.alpha=30 --agent.encoder=impala_small --agent.dataset.p_aug=0.5 --agent.dataset.frame_stack=3

# visual scene play
python main.py --env_name=visual-scene-play-v0 --agent=agents/byol_gamma.py --agent.batch_size=256 --agent.alpha=3 --agent.normalize_q_loss=True --agent.encoder=impala_small --agent.dataset.p_aug=0.5 --agent.dataset.frame_stack=3
```

</details>

### ICVF

<details>
<summary><b>Click to expand the full list of commands</b></summary>

```shell
# exorl walker
python main.py --env_name=exorl-rnd-walker --agent=agents/icvf.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.alpha=0.03

# exorl cheetah
python main.py --env_name=exorl-rnd-cheetah --agent=agents/icvf.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01

# exorl quadruped
python main.py --env_name=exorl-rnd-quadruped --agent=agents/icvf.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.alpha=0.03

# exorl jaco
python main.py --env_name=exorl-rnd-jaco --agent=agents/icvf.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.alpha=0.03

# antmaze large navigate
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/icvf.py --agent.alpha=3

# antmaze teleport navigate
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/icvf.py

# cube single play
python main.py --env_name=ogbench-cube-single-play-v0 --agent=agents/icvf.py --agent.alpha=30

# scene play
python main.py --env_name=ogbench-scene-play-v0 --agent=agents/icvf.py
```

</details>

### HILP

<details>
<summary><b>Click to expand the full list of commands</b></summary>

```shell
# exorl walker
python main.py --env_name=exorl-rnd-walker --agent=agents/hilp.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.expectile=0.9 --agent.alpha=0.3

# exorl cheetah
python main.py --env_name=exorl-rnd-cheetah --agent=agents/hilp.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.alpha=3

# exorl quadruped
python main.py --env_name=exorl-rnd-quadruped --agent=agents/hilp.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.expectile=0.9 --agent.alpha=3

# exorl jaco
python main.py --env_name=exorl-rnd-jaco --agent=agents/hilp.py --agent.repr_hidden_dims="(512,512)" --agent.value_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.expectile=0.9 --agent.alpha=0.3

# antmaze large navigate
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/hilp.py --agent.normalize_q_loss=True

# antmaze teleport navigate
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/hilp.py --agent.normalize_q_loss=True

# cube single play
python main.py --env_name=ogbench-cube-single-play-v0 --agent=agents/hilp.py --agent.normalize_q_loss=True

# scene play
python main.py --env_name=ogbench-scene-play-v0 --agent=agents/hilp.py --agent.expectile=0.9 --agent.normalize_q_loss=True

# visual cube single play
python main.py --env_name=visual-cube-single-play-v0 --agent=agents/hilp.py --agent.batch_size=256 --agent.normalize_q_loss=True --agent.encoder=impala_small --agent.dataset.p_aug=0.5 --agent.dataset.frame_stack=3

# visual scene play
python main.py --env_name=visual-scene-play-v0 --agent=agents/hilp.py --agent.batch_size=256 --agent.alpha=3 --agent.normalize_q_loss=True --agent.encoder=impala_small --agent.dataset.p_aug=0.5 --agent.dataset.frame_stack=3
```

</details>

### FB

<details>
<summary><b>Click to expand the full list of commands</b></summary>

```shell
# exorl walker
python main.py --env_name=exorl-rnd-walker --agent=agents/fb.py --agent.repr_hidden_dims="(256,256,256)" --agent.forward_repr_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.orthonorm_coeff=0.03 --agent.alpha=0 --agent.tanh_squash=False

# exorl cheetah
python main.py --env_name=exorl-rnd-cheetah --agent=agents/fb.py --agent.repr_hidden_dims="(256,256,256)" --agent.forward_repr_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.orthonorm_coeff=0.3 --agent.alpha=0 --agent.tanh_squash=False

# exorl quadruped
python main.py --env_name=exorl-rnd-quadruped --agent=agents/fb.py --agent.repr_hidden_dims="(256,256,256)" --agent.forward_repr_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.orthonorm_coeff=1 --agent.alpha=0

# exorl jaco
python main.py --env_name=exorl-rnd-jaco --agent=agents/fb.py --agent.repr_hidden_dims="(256,256,256)" --agent.forward_repr_hidden_dims="(1024,1024,1024)" --agent.actor_hidden_dims="(1024,1024,1024)" --agent.activation=relu --agent.latent_dim=50 --agent.discount=0.98 --agent.tau=0.01 --agent.orthonorm_coeff=0.3 --agent.alpha=0

# antmaze large navigate
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/fb.py --agent.alpha=0.03 --agent.normalize_q_loss=True

# antmaze teleport navigate
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/fb.py --agent.reward_temperature=3 --agent.alpha=0.03 --agent.normalize_q_loss=True

# cube single play
python main.py --env_name=ogbench-cube-single-play-v0 --agent=agents/fb.py --agent.reward_temperature=10 --agent.normalize_q_loss=True

# scene play
python main.py --env_name=ogbench-scene-play-v0 --agent=agents/fb.py --agent.reward_temperature=3 --agent.normalize_q_loss=True

# visual cube single play
python main.py --env_name=visual-cube-single-play-v0 --agent=agents/fb.py --agent.batch_size=256 --agent.reward_temperature=300 --agent.alpha=1 --agent.normalize_q_loss=True --agent.encoder=impala_small --agent.dataset.p_aug=0.5 --agent.dataset.frame_stack=3

# visual scene play
python main.py --env_name=visual-scene-play-v0 --agent=agents/fb.py --agent.batch_size=256 --agent.reward_temperature=300 --agent.encoder=impala_small --agent.dataset.p_aug=0.5 --agent.dataset.frame_stack=3
```

</details>

# Acknowledgements

This codebase is adapted from [OGBench](https://github.com/seohongpark/ogbench) and [FQL](https://github.com/seohongpark/fql) implementations. References for baseline implementation include [Meta Motivo](https://github.com/facebookresearch/metamotivo), and [HILP](https://github.com/seohongpark/HILP).

# Citation

```bibtex
@article{zheng2026can,
  title={Can We Really Learn One Representation to Optimize All Rewards?}, 
  author={Zheng, Chongyi and Jayanth, Royina Karegoudra and Eysenbach, Benjamin},
  journal={arXiv preprint arXiv:2602.11399},
  year={2026},
}
```
