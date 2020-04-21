import numpy as np
import matplotlib.pyplot as plt


def generate_sample(loaders):
    train_imgs = next(iter(loaders))
    print('Shape of Grayscale Image Tensor is : {}'.format(train_imgs[0].shape))
    print('Shape of Colored Image Tensor is : {}'.format(train_imgs[1].shape))
    train_img_to_show_bw = train_imgs[0][0][0]
    train_img_to_show_rgb = train_imgs[1][0]
    train_img_to_show_rgb = np.transpose((train_img_to_show_rgb + 1) / 2, (1, 2, 0))

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(train_img_to_show_rgb)
    axes[1].imshow(train_img_to_show_bw, cmap='gray')
    plt.show()

def generate_loss_chart(labels, train, test):
    n = np.arange(len(labels))
    width = 0.3

    if len(labels) != len(train) != len(test):
        raise Exception('Length of the arrays do not match!')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.bar(n, train, width, color='g', label='Loss over training set')
    ax.bar(n + width, test, width, color='y', label='Loss over test set')

    ax.set_ylabel('Loss')
    ax.set_xticks(n + width / 2)
    ax.set_xticklabels(labels)

    ax.legend()

    plt.savefig('compare_loss.png')

    plt.show()