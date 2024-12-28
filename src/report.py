
from matplotlib.pylab import plt
import csv


def plot_train_loss_graph(steps, losses, file_path):
    plt.clf()
    plt.plot(steps, losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig(file_path)


def plot_total_rewards_graph(episodes, total_rewards, file_path):
    plt.clf()
    plt.plot(episodes, total_rewards, label='Total rewards per episode')
    plt.title('Total rewards per episode')
    plt.xlabel('Episode')
    plt.ylabel('Total rewards')
    plt.savefig(file_path)


def render_interactive_plot(csv_file_path: str):
    """
        csv_file_path = '2024-12-11T05_02_13.386597+00_00-step_loss.csv'
        render_interactive_plot(csv_file_path)
    """
    x = []
    y = []

    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            x.append(int(row[0]))
            y.append(float(row[1]))

    plt.plot(x, y, color='g', linestyle='dashed',
             marker='o', label="Training Criterion")

    plt.xticks(rotation=25)
    plt.xlabel('Step')
    plt.ylabel('Criterion')
    plt.title('Criterion Report', fontsize=20)
    plt.grid()
    plt.legend()
    plt.show()

