#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# test mae crop
# ./.python-greene submitit_hydra.py $comp exp=mae_crop name="$(date +%F)-train_1GPU_testing2"

# test mae dino
# ./.python-greene submitit_hydra.py $comp exp=mae_dino name="$(date +%F)-train_1GPU_testing_10k4x4x"

# test random proximal dino
./.python-greene submitit_hydra.py $comp exp=random_proximal_crop_dino name="$(date +%F)-train_1GPU_rand_prox_dino"

# eval dino
# ./.python-greene submitit_hydra.py $comp exp=eval_dino name="$(date +%F)-train_1GPU_nocrop"
