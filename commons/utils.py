import matplotlib.pyplot as plt


def plot_figure(data, xlabel, ylabel, filename):
    plt.figure()
    plt.cla()
    plt.plot(range(len(data)), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{filename}", format="png")
    plt.close()
