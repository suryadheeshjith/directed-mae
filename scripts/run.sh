#!/bin/bash

# Obvious / static arguments

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# mae pretrain
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000 exp=main_pretrain name="$(date +%F)-atest"
# dino
# ./.python-greene submitit_hydra.py compute/greene=1x4 compute/greene/node=rtx8000 exp=main_dino name="$(date +%F)-train_4GPU_1"

# mae crops
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000 exp=mae_crop name="$(date +%F)-1GPU_save_crops_100004x4x"

# mae dino
# 4 GPU
# ./.python-greene submitit_hydra.py compute/greene=1x4 compute/greene/node=rtx8000 exp=mae_dino name="$(date +%F)-4GPU_800_crop4x4x"

# 1 GPU - 2 hrs
./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=mae_dino name="$(date +%F)-testing"

# dino eval
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=eval_dino name="$(date +%F)-780_4x4x"