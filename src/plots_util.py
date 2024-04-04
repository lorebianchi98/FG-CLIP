import matplotlib.pyplot as plt
import torch

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def plot_values(values, labels, ylabel, title, warmup=False):
    start = 0 if warmup else 1
    n_epochs = len(values[0])
    epochs = range(start, n_epochs)

    for value, label in zip(values, labels):
        linestyle = 'solid' if label != labels[-1] else 'dashed'
        plt.plot(epochs, value[start:], label=label, linestyle=linestyle)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    plt.show()

def plot_loss(losses, n_col=20):
    n = losses.shape[0]
    normalized_loss = torch.mean(losses[:n - (n % n_col)].view(n_col, -1), dim=0)
    normalized_loss.shape
    plt.plot(normalized_loss)
    plt.xlabel('Iteration')
    plt.ylabel("loss")
    
def plot_losses_unnormalized(train_losses, val_losses, additional_val_losses=None, labels=["Training Loss", "Validation Loss", 'Validation Loss 2nd set'], plot=False, path=None):
    epochs = range(0, len(train_losses))

    losses = [train_losses, val_losses] if additional_val_losses is None else [train_losses, val_losses, additional_val_losses]
    
    for loss, label in zip(losses, labels):    
        plt.plot(epochs, loss, label=label)

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if path:
        plt.savefig(path)
    if plot:
        plt.show()

def plot_losses(train_losses, val_losses, additional_val_losses=None, labels=["Training Loss", "Validation Loss", 'Validation Loss 2nd set'], warmup=False, plot=False, path=None):
    val_start = 0 if warmup else 1
    epochs_train = range(1, len(val_losses)) 
    epochs_val = range(val_start, len(val_losses))

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    line1, = ax1.plot(epochs_train, train_losses[1:], color=color, label=labels[0])
    ax1.tick_params(axis='y')

    color = 'tab:orange'
    line2, = ax1.plot(epochs_val, val_losses[val_start:], color=color, label=labels[1])
    if additional_val_losses is not None:
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.spines['right'].set_position(('outward', 0))  # offset the third axis
        ax2.set_ylabel(labels[2], color=color)
        line3, = ax2.plot(epochs_val, additional_val_losses[val_start:], color=color, label=labels[2])
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(False)

    lines = [line1, line2, line3] if additional_val_losses is not None else [line1, line2]
    labels = [line.get_label() for line in lines]

    ax1.legend(lines, labels, loc='upper right')
    plt.title('Training and Validation Loss')
    if path:
        plt.savefig(path)
    if plot:
        plt.show()