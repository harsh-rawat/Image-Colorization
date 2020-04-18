import numpy as np
import matplotlib.pyplot as plt


def generate_sample(loaders):
    train_imgs = next(iter(loaders))
    print(train_imgs[0].shape)
    print(train_imgs[1].shape)
    train_img_to_show_bw = train_imgs[0][0][0]
    train_img_to_show_rgb = train_imgs[1][0]
    train_img_to_show_rgb = np.transpose((train_img_to_show_rgb + 1) / 2, (1, 2, 0))

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(train_img_to_show_rgb)
    axes[1].imshow(train_img_to_show_bw, cmap='gray')
    plt.show()
