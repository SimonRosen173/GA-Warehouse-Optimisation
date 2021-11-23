import wandb
from wandb import Api
import pickle
import numpy as np
from Visualisation.vis import VisGrid


def main():
    api = wandb.Api()
    run = api.run("simonrosen42/GARuck/2p1ekv6x")
    for file in run.files():
        print(file)


def test():
    with open("pops/pop_5.pkl", "rb") as f:
        pop_data = list(pickle.load(f))
    grid = pop_data[0][1].tolist()
    vis_grid = VisGrid(grid, (900, 400))
    vis_grid.window.getMouse()
    vis_grid.save_to_png("imgs/grid_real")
    print(grid)


if __name__ == "__main__":
    # main()
    test()
