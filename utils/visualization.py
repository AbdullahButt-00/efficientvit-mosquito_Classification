import matplotlib.pyplot as plt
import torchvision

def show_batch(images, labels, idx2label, title="Batch"):
    images = images[:8]  # show only 8 images
    labels = labels[:8]

    grid_img = torchvision.utils.make_grid(images, nrow=4, normalize=True)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(f"{title} - Labels: {[idx2label[int(label)] for label in labels]}")
    plt.axis('off')
    plt.show()
