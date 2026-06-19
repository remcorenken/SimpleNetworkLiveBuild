import numpy as np
import struct
import matplotlib.pyplot as plt

import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def GenSinTrainSet(x: np.array,s:float) -> np.array:
    x=np.asarray(x)
    y=np.sin(x)
    ye=y+np.random.randn(*x.shape)*s
    return y,ye
    
def get_valid_folder() -> str:
    path1 = r"C:\Users\RenkenRJ\PyCharmMiscProject\TrainDataMNIST" #modify for your local system(s)
    path2 = r"C:\Users\remco\PycharmProjects\SimpleNetworkLiveBuild\TrainDataMNIST"

    # 1. Try first file
    if os.path.exists(path1):
        return path1

    # 2. Try second file
    if os.path.exists(path2):
        return path2


    # 3. If both fail, ask the user via GUI
    print("Neither file found. Please select a file.")

    # Hide the root Tk window
    root = Tk()
    root.withdraw()

    filename = askopenfilename(
        title="Select your data file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if filename:
        return filename
    else:
        raise FileNotFoundError("No file selected and default files not found.")

def read_idx(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        # Read the magic number and dimensions
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        # Read the remaining data
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
        return data

if __name__ == '__main__':# Usage:
    train_images = read_idx(os.path.join(get_valid_folder(), 'train-images.idx3-ubyte'))
    train_labels = read_idx(os.path.join(get_valid_folder(), 'train-labels.idx1-ubyte'))


    print(train_images.shape)  # (60000, 28, 28)
    print(train_labels.shape)  # (60000,)


    plt.imshow(train_images[0], cmap='gray')
    plt.title(f"Label: {train_labels[0]}")
    plt.show()