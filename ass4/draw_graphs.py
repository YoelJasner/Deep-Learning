STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}

import matplotlib.pyplot as plt
import json
import yaml


def plot_graphs(filename, task):
    results = []
    for result in open(filename):
        results.append(yaml.safe_load(result))

    epochs = results[0]['epochs']
    accus = [100 * a for a in results[0]['result']]


    _, ax = plt.subplots()
    for i,result in enumerate(results):
        ax.plot(epochs, [100 * a for a in result['result']], label=result['name'])
    ax.set_ylabel('accuracy (%)')
    ax.set_xlabel('epochs')
    ax.grid(True)
    ax.set_title(task)
    ax.legend(loc='lower right', fontsize="x-large")
    plt.draw()


if __name__ == "__main__":
    plot_graphs("dev_accus.json", "BiLSTM max-pooling")
    #plot_graphs("dev_loss.json", "BiLSTM max-pooling")
    plt.show()
