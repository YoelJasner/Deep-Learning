STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}

import matplotlib.pyplot as plt
import json
import yaml

def plot_graphs(filename, task):
    results = []
    for result in open(filename):
        results.append(yaml.safe_load(result))

    sentences_parts = results[0]['sentences']
    sentences_parts = [p / 100 for p in sentences_parts]

    colors = ['r', 'g', 'b', 'y']

    _, ax = plt.subplots()
    for i,result in enumerate(results):
        ax.plot(sentences_parts, result['dev_acc'], colors[i], label=result['name'])
    ax.set_ylabel('accuracy (%)')
    ax.set_xlabel('number of sentences seen / 100')
    ax.grid(True)
    ax.set_title(task + " - accu for different options")
    ax.legend(loc='center right', fontsize="x-large")
    plt.draw()


if __name__ == "__main__":
    plot_graphs("ner_dev_results.json", "NER")
    plot_graphs("pos_dev_results.json", "POS")
    plt.show()