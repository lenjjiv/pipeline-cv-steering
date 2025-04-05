from PIL import Image
import matplotlib.pyplot as plt


def show_image(image):
    """Отображение одного изображения"""
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def compare_images(images, titles=None, figsize=(15, 8)):
    """Отображение нескольких изображений для сравнения"""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for i, img in enumerate(images):
        axes[i].imshow(img)
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()