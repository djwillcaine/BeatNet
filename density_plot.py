import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import median, mean

def density_plot(save_path):
    plt.figure(figsize=(8, 3.75))
    plt.title('Density Plots')
    plt.xlabel('BPM')
    plt.ylabel('Density')

    for ds in os.listdir('data'):
        subdir = os.path.join('data', ds, 'training')
        data = []
        for bpm in os.listdir(subdir):
            for _ in os.listdir(os.path.join(subdir, str(bpm))):
                data.append(int(bpm))
        label = '%s (median=%d, mean=%d)' % (ds, median(data), mean(data))
        sns.distplot(data, label=label, hist=False)

    plt.legend()
    plt.savefig(save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--save-path', default='density_plots.png',
        help='File path to save the resulting plot.')
    args = parser.parse_args()

    density_plot(args.save_path)