import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_grad_flow(named_parameters, epoch, out_dir):
    """Plots the gradients flowing through different layers
    in the net during training. Can be used for checking for
    possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after
    loss.backward() as  "plot_grad_flow(self.model.named_parameters())"
    to visualize the gradient flow.
    """

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(1, len(max_grads) + 1), max_grads,
            alpha=0.1, width=0.5, color="c")
    plt.bar(np.arange(1, len(max_grads) + 1), ave_grads,
            alpha=0.1, width=0.5, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(1, len(ave_grads) + 1, 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads) + 1)
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title(f"Epoch {epoch}. Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)],
               ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, f'epoch_{epoch}_gf.png'))


def draw_plots(epochs, plots_path, prefix,
               loss_train, loss_val, acc_val):

        time_str = datetime.now().strftime('%Y-%m-%d%H-%M-%S')

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(1, epochs + 1), loss_train,
                 color='blue', label="train_loss")
        plt.plot(np.arange(1, epochs + 1), loss_val,
                 color='red', label="test_loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(plots_path,
                                 "%s_loss_%s.png" % (prefix, time_str)),
                    dpi=300)
        plt.close()

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(1, epochs + 1), acc_val,
                 color='red', label="test_metric")
        plt.title("Metric")
        plt.xlabel("Epoch #")
        plt.ylabel("Jaccard index")
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(plots_path,
                                 "%s_metric_%s.png" % (prefix, time_str)),
                    dpi=300)
        plt.close()


def get_model_name(model):
    r = model.__repr__()
    return r[:r.find('(')]