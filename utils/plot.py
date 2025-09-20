import struct
import gzip
import numpy as np
import matplotlib.pyplot as plt

def read_idx_images(filename):
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(filename, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        num_images, rows, cols = struct.unpack(">III", f.read(12))
        
        # Read the rest as unsigned bytes
        data = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape to (num_images, rows, cols)
        return data.reshape(num_images, rows, cols)

def read_idx_labels(filename):
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(filename, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        num_items, = struct.unpack(">I", f.read(4))
        
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

def plot(imageFilePath, imageLabelPath, index):
    images = read_idx_images(imageFilePath)
    labels = read_idx_labels(imageLabelPath)

    print("Images shape:", images.shape)   # (60000, 28, 28)
    print("Labels shape:", labels.shape)   # (60000,)
    plt.imshow(images[index], cmap="gray")
    plt.title(f"Label: {labels[0]}")
    plt.axis("off")
    plt.show()

imageFilePath = "data/FashionMNIST/raw/t10k-images-idx3-ubyte"
imageLabelPath = "data/FashionMNIST/raw/t10k-labels-idx1-ubyte"
plot(imageFilePath, imageLabelPath, 9998)