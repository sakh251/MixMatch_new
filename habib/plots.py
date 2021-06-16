import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib

def smooth(scalars, smooth_factor):
    """
    Smoothing by exponential moving average
    :param scalars:
    :param smooth_factor:
    :return: smoothed scalars
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * smooth_factor + (1 - smooth_factor) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.array(smoothed)

def plot_one_curve(ax, csv_path, color, label, key, smoth):
    df = pd.read_csv(csv_path, nrows=1000)
    x, y = df['step'], df[key]
    y = smooth(y, smoth)
    ax.plot(x, y, color=color, label=label,  )

    return ax

def plot_top1_validation(paths, colors, legends, outdir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set(xlabel='Steps',
           ylabel='Validation Accuracy')
    ax.set_xlabel('Steps',fontsize=15)
    ax.set_ylabel('Validation Accuracy', fontsize=15)
    ax.grid()
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 9}
    plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)  # fontsize of the tick labels
    plt.ylim(ymax=1, ymin=0.9)
    matplotlib.rc('font', **font)
    for i in range(len(paths)):
        # if i == 0:
        #     smoth = 0
        # else:
        smoth=0.90
        plot_one_curve(ax, paths[i], colors[i], legends[i], 'validationAcc', smoth)

    plt.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join('figures', outdir, 'top1_validation_80.pdf'))
    fig.savefig(os.path.join('figures', outdir, 'top1_validation_80.png'), dpi=1000)
    plt.show()


def plot_top1_validationLoss(paths, colors, legends, outdir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set(xlabel='Steps',
           ylabel='Validation Accuracy')
    ax.set_xlabel('Steps',fontsize=15)
    ax.set_ylabel('Validation Accuracy', fontsize=15)
    ax.grid()
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 9}
    plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)  # fontsize of the tick labels
    # plt.ylim(ymax=1, ymin=0.9)
    matplotlib.rc('font', **font)
    for i in range(len(paths)):
        # if i == 0:
        #     smoth = 0
        # else:
        smoth=0.90
        plot_one_curve(ax, paths[i], colors[i], legends[i], 'validationLoss', smoth)

    plt.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join('figures', outdir, 'validationLoss.pdf'))
    fig.savefig(os.path.join('figures', outdir, 'validationLoss.png'), dpi=1000)
    plt.show()


def plots():
    outdir = '.'
    # n_gpus = np.array([1, 2, 3, 4, 6])
    # legends = ['adhoc CNN', 'VGG16', 'ResNet101', 'Resnet50', 'WideResNet50']
    # throughput = np.array([7368.07, 2539.62, 2332.10, 3546.89, 2096.33])
    # plot_throughput_arch(legends, throughput, outdir)
    #
    # train_time = np.array([1719.86, 3322.08, 3322.91, 2478.02, 3838.02 ])
    # plot_training_time_arch(legends, train_time, outdir)

    colors = ['orange', 'blue',  'purple', 'brown', 'gray']
    legends = ['HH,HV', 'HH,HV,IA', 'HH,HV,IA,NOISE']
    # path = "/home/skh018/"
    path = "models/backup/"

    top1_train_paths = [path + 'model_cnn_13layer_batch_100_dataset_seaice_channels_HH_HV_3_aug_logs.txt',
                        path + 'model_cnn_13layer_batch_100_dataset_seaice_channels_HH_HV_IA_4_logs.txt',
                        path + 'model_cnn_13layer_batch_100_dataset_seaice_channels_HH_HV_IA_noise_4_logs.txt',
                        ]


    plot_top1_validation(top1_train_paths, colors, legends, outdir)

if __name__ == '__main__':
    plots()

