# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     gpuType: A100
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 09 · Part 2: train our own LeWorldModel on Push-T (Colab A100)
#
# Trains LeWM with stable-worldmodel's own recipe, so the checkpoint loads
# straight into the planner of Part 1 (`plan.py`, Docker on a laptop).
# ~3.5 h on an A100 for 3 short epochs (13,791 steps); the bottleneck is
# reading frames from the 46 GB HDF5 file, not the GPU.

# %%
import os
import subprocess

SWM_COMMIT = "21446f1ede6d5284e981bd7b47f432b994e6d812"
HOME = "/content/swmhome"
os.environ["STABLEWM_HOME"] = HOME
sh = lambda cmd: subprocess.run(cmd, shell=True, check=True)

sh("apt-get -qq install -y zstd > /dev/null")
sh(f'pip install -q "stable-worldmodel[train] @ git+https://github.com/galilai-group/stable-worldmodel@{SWM_COMMIT}" '
   '"transformers>=4.50,<5" h5py hdf5plugin')
sh(f"rm -rf swm && git clone -q https://github.com/galilai-group/stable-worldmodel swm && git -C swm checkout -q {SWM_COMMIT}")

# %% [markdown]
# ## Data: LeWM's Push-T expert set (13 GB download, 46 GB unpacked)

# %%
from huggingface_hub import hf_hub_download

os.makedirs(f"{HOME}/datasets", exist_ok=True)
zst = hf_hub_download("quentinll/lewm-pusht", "pusht_expert_train.h5.zst", repo_type="dataset",
                      local_dir=f"{HOME}/datasets")
sh(f"zstd -d -q --rm {zst} -o {HOME}/datasets/pusht_expert_train.h5")
sh(f"cat {HOME}/datasets/pusht_expert_train.h5 > /dev/null")     # load into the page cache: random reads get faster

# %% [markdown]
# ## Train
#
# Official defaults (ViT-tiny encoder, 6-layer predictor, SIGReg weight
# 0.09, batch 128, lr 5e-5) except: 3 epochs of 33% of the data each,
# 11 loader workers. A checkpoint is saved after every epoch under
# `$STABLEWM_HOME/checkpoints/lewm/`.

# %%
sh("cd swm/scripts/train && python -u lewm.py "
   f"data.dataset.name={HOME}/datasets/pusht_expert_train.h5 wandb.enabled=False "
   "num_workers=11 trainer.max_epochs=3 +trainer.limit_train_batches=0.33")
sh(f"ls -la {HOME}/checkpoints/lewm")

# %% [markdown]
# Copy `config.json` + `weights_epoch_3.pt` into a folder under
# `$STABLEWM_HOME/checkpoints/` on the laptop (the Docker volume), then:
# `docker run ... jepa-wm --policies <folder name>`.
