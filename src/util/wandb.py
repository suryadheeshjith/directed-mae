import os

import wandb
from util.image import convert_to_pil
import logging

def init_wandb(cfg, name, dir, project="directed-mae"):
    wandb.login(key=os.environ["WANDB_KEY"])
    return wandb.init(project=project, 
                      config=cfg,
                      name=name,
                      dir=dir)

def get_wandb_image(val, caption=None, mode="tensor"):
    # Matplotlib plot
    if mode == "plt":
        return wandb.Image(val, caption)
    
    # Tensor
    return wandb.Image(convert_to_pil(val), caption=caption)

def get_wandb_plot(data, columns, title):
    table = wandb.Table(columns=columns, data=data)
    return wandb.plot.line(table, columns[0], columns[1], title=title)
