import numpy as np
import struct
import matplotlib.pyplot as plt

def read_idx(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        # Read the magic number and dimensions
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        # Read the remaining data
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
        return data

if __name__ == '__main__':# Usage:
    train_images = read_idx(r"C:\Users\RenkenRJ\PyCharmMiscProject\TrainDataMNIST\train-images.idx3-ubyte")
    train_labels = read_idx(r"C:\Users\RenkenRJ\PyCharmMiscProject\TrainDataMNIST\train-labels.idx1-ubyte")


    print(train_images.shape)  # (60000, 28, 28)
    print(train_labels.shape)  # (60000,)


    plt.imshow(train_images[0], cmap='gray')
    plt.title(f"Label: {train_labels[0]}")
    plt.show()