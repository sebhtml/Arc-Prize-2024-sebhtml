
from matplotlib.pylab import plt


def plot_train_loss_graph(steps, losses, file_path):
    plt.plot(steps, losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig(file_path)
